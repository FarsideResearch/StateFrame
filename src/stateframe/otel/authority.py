"""Fail-closed admission for local telemetry source and process policy."""
from __future__ import annotations

import grp
import json
import os
from pathlib import Path
import pwd
import stat
import sys
from typing import Mapping, TYPE_CHECKING
from urllib.parse import unquote

from ..engine.kernel_safe_paths import safe_home
from .types import Coverage, Reason

if TYPE_CHECKING:
    from .transport import TransportSettings


MAX_AUTHORITY_BYTES = 1024 * 1024
_SENSITIVE_FRAGMENTS = ("API_KEY", "APIKEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "AUTHORIZATION", "CREDENTIAL")


def admit(root: Path, view) -> Coverage:
    """Check ownership and reject source authority over exporter settings."""
    if not _private_root(root) or not _validate_authority_records(root, view):
        return _coverage(Reason.SOURCE_UNTRUSTED)
    if _has_secret_alias(view.environment):
        return _coverage(Reason.UNSAFE_ALIAS)
    spec = _read_record(root / "spec.json")
    if spec is None or not _consistent_spec(spec, view):
        return _coverage(Reason.SOURCE_UNTRUSTED)
    if _overlapping_grants(root, spec, view.grants):
        return _coverage(Reason.ISOLATION_UNVERIFIED)
    return Coverage(True, ())


def application_credentials(env: Mapping[str, str], spec: dict[str, object] | None = None) -> tuple[str, ...]:
    spec = spec or {}
    routes = spec.get("routes")
    roles = routes.get("roles") if type(routes) is dict else None
    names = [spec.get("api_key_env")]
    if type(roles) is dict:
        names.extend(role.get("api_key_env") for role in roles.values() if type(role) is dict)
    aliases = {name for name in names if type(name) is str}
    return tuple({value for name, value in env.items()
                  if isinstance(name, str) and not name.startswith("OTEL_") and isinstance(value, str) and value
                  and (name in aliases or any(part in name.upper() for part in _SENSITIVE_FRAGMENTS))})


def has_application_alias(env: Mapping[str, str], settings: TransportSettings | None = None,
                          credentials: tuple[str, ...] = ()) -> bool:
    credentials = credentials or application_credentials(env)
    if not credentials:
        return False
    fields = () if settings is None else (settings.endpoint, *(item for pair in settings.headers for item in pair))
    raw = (value for name, value in env.items()
           if isinstance(name, str) and name.startswith("OTEL_EXPORTER_OTLP_") and isinstance(value, str))
    for value in (*fields, *raw):
        decoded = unquote(value)
        if any(secret in value or secret in decoded for secret in credentials):
            return True
    return False


def config_path_allowed(path: Path | None, root: Path | None = None, grants: tuple[Path, ...] = ()) -> bool:
    if path is None:
        return True
    try:
        info = path.lstat()
        resolved = path.resolve(strict=True)
        run_root = root.resolve(strict=True) if root is not None else None
    except OSError:
        return False
    if (not path.is_absolute() or not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
            or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) & 0o077
            or info.st_size > MAX_AUTHORITY_BYTES or not _safe_parent_chain(path)):
        return False
    if resolved != path:
        return False
    if run_root is not None and _overlap(resolved, run_root):
        return False
    home = safe_home()
    if home is None:
        return False
    state_root = home / ".stateframe-otel"
    if _overlap(resolved, state_root):
        return False
    spec = _read_record(run_root / "spec.json") if run_root is not None else None
    repo = spec.get("repo") if spec is not None else None
    protected = (*grants, Path(sys.prefix), Path(sys.base_prefix))
    if type(repo) is str:
        protected += (Path(repo),)
    for grant in protected:
        if not isinstance(grant, Path) or not grant.is_absolute() or _overlap(resolved, grant.resolve(strict=False)):
            return False
    return True


def transport_paths_allowed(settings, root: Path | None = None, grants: tuple[Path, ...] = ()) -> bool:
    for name in ("ca", "cert", "key_file"):
        path = getattr(settings, name, None)
        if path is not None and not config_path_allowed(path, root, grants):
            return False
    return True


def source_mode_problem(root: Path) -> tuple[Path, str] | None:
    """Return one permission repair for the operator's explicitly selected path."""
    for directory in (*reversed(root.parents), root):
        try:
            info = directory.lstat()
        except OSError:
            return None
        if not stat.S_ISDIR(info.st_mode):
            return None
        mode = stat.S_IMODE(info.st_mode)
        if directory == root:
            if info.st_uid == os.getuid() and mode & 0o077:
                return directory, "700"
            if (mode & 0o022
                    and not (directory in {Path("/tmp"), Path("/var/tmp")} and mode & stat.S_ISVTX)):
                return directory, "go-w"
            continue
        if unsafe_parent_mode(directory, info):
            return directory, "go-w"
    return None


def _private_root(root: Path) -> bool:
    if not isinstance(root, Path) or not root.is_absolute():
        return False
    try:
        original = root.lstat()
        resolved = root.resolve(strict=True)
    except OSError:
        return False
    if not stat.S_ISDIR(original.st_mode) or original.st_uid != os.getuid() or stat.S_IMODE(original.st_mode) & 0o077:
        return False
    if root != resolved:
        return False
    current = resolved
    while True:
        try:
            info = current.lstat()
        except OSError:
            return False
        if stat.S_ISLNK(info.st_mode):
            return False
        if current != resolved and unsafe_parent_mode(current, info):
            return False
        parent = current.parent
        if parent == current:
            break
        current = parent
    return True


def _validate_authority_records(root: Path, view) -> bool:
    expected = {root / "spec.json"}
    settings = root / "tool-settings.json"
    if settings.exists():
        expected.add(settings)
    try:
        records = set(view.authority_records)
    except (AttributeError, TypeError):
        return False
    return records == expected and all(_private_file(path) for path in expected)


def _private_file(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    return (stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid() and info.st_nlink == 1
            and not info.st_mode & stat.S_IWOTH
            and (not info.st_mode & stat.S_IWGRP or _private_root(path.parent))
            and info.st_size <= MAX_AUTHORITY_BYTES)


def _safe_parent_chain(path: Path) -> bool:
    current = path.parent
    while True:
        try:
            info = current.lstat()
        except OSError:
            return False
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            return False
        if unsafe_parent_mode(current, info):
            return False
        if current.parent == current:
            return True
        current = current.parent


def unsafe_parent_mode(directory: Path, info) -> bool:
    mode = stat.S_IMODE(info.st_mode)
    if not mode & 0o022:
        return False
    if directory in {Path("/tmp"), Path("/var/tmp")} and mode & stat.S_ISVTX:
        return False
    if mode & stat.S_IWOTH:
        return True
    return bool(mode & stat.S_IWGRP) and not _private_primary_group(info.st_gid)


def _private_primary_group(gid: int) -> bool:
    """Return whether group write is confined to the current account."""
    try:
        uid = os.getuid()
        account = pwd.getpwuid(uid)
        username = account.pw_name
        primary_gid = account.pw_gid
        if (type(username) is not str or not username or type(primary_gid) is not int
                or gid != primary_gid):
            return False

        group = grp.getgrgid(primary_gid)
        if (type(group.gr_gid) is not int or group.gr_gid != primary_gid
                or type(group.gr_mem) not in {list, tuple}
                or any(type(member) is not str or member != username for member in group.gr_mem)):
            return False

        accounts = pwd.getpwall()
        if type(accounts) not in {list, tuple}:
            return False
        current_account_found = False
        for candidate in accounts:
            candidate_name = candidate.pw_name
            candidate_uid = candidate.pw_uid
            candidate_gid = candidate.pw_gid
            if (type(candidate_name) is not str or type(candidate_uid) is not int
                    or type(candidate_gid) is not int):
                return False
            if candidate_name == username and candidate_uid == uid and candidate_gid == primary_gid:
                current_account_found = True
            elif candidate_gid == primary_gid:
                return False
        return current_account_found
    except Exception:
        return False


def _read_record(path: Path) -> dict[str, object] | None:
    if not _private_file(path):
        return None
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, flags)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > MAX_AUTHORITY_BYTES:
                return None
            chunks: list[bytes] = []
            remaining = MAX_AUTHORITY_BYTES + 1
            while remaining:
                block = os.read(fd, min(remaining, 64 * 1024))
                if not block:
                    break
                chunks.append(block)
                remaining -= len(block)
            raw = b"".join(chunks)
            if len(raw) > MAX_AUTHORITY_BYTES:
                return None
        finally:
            os.close(fd)
        value = json.loads(raw.decode("utf-8", "strict"), object_pairs_hook=_unique_pairs,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        return value if type(value) is dict else None
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError, RecursionError):
        return None


def _consistent_spec(spec: dict[str, object], view) -> bool:
    if spec.get("schema") != "stateframe-run/1":
        return False
    seed = spec.get("run_identity_seed")
    if not isinstance(seed, str) or len(seed) != 32 or any(ch not in "0123456789abcdef" for ch in seed):
        return False
    try:
        if bytes.fromhex(seed) != view.seed:
            return False
    except ValueError:
        return False
    if spec.get("profile") != view.profile:
        return False
    resume = spec.get("resume")
    attempt = resume.get("attempt", 1) if type(resume) is dict else 1
    if type(attempt) is not int or attempt != view.attempt:
        return False
    created = spec.get("created")
    if not isinstance(created, str) or len(created) > 64:
        return False
    repo = spec.get("repo")
    if (type(repo) is not str or not repo or len(repo) > 4096
            or not Path(repo).is_absolute() or Path(repo).resolve(strict=False) != Path(repo)):
        return False
    if "otel" in spec:
        return False
    return True


def _overlapping_grants(root: Path, spec: dict[str, object], grants: tuple[Path, ...]) -> bool:
    repo_value = spec.get("repo")
    repo = Path(repo_value) if isinstance(repo_value, str) else None
    for grant in grants:
        if not isinstance(grant, Path) or not grant.is_absolute():
            return True
        candidate = grant.resolve(strict=False)
        if _overlap(root, candidate) or repo is not None and _overlap(repo.resolve(strict=False), candidate):
            return True
    if repo is not None and repo.is_absolute() and _is_relative(root, repo.resolve(strict=False)):
        return True
    return False


def _overlap(left: Path, right: Path) -> bool:
    return _is_relative(left, right) or _is_relative(right, left)


def _is_relative(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _has_secret_alias(environment: Mapping[str, object]) -> bool:
    return any(type(name) is not str or name.startswith("OTEL_") and value is True
               for name, value in environment.items())


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError
        result[key] = value
    return result


def _coverage(*reasons: Reason) -> Coverage:
    unique = tuple(sorted({(reason, 1) for reason in reasons}, key=lambda item: item[0].value))
    return Coverage(False, unique)
