"""Private, installation-scoped authentication bindings for request measurements."""
from __future__ import annotations

import hashlib
import hmac
import os
from pathlib import Path
import secrets
import stat
import sys
import threading

if __package__:
    from .kernel_reason_log import builtin_exception_name
else:
    from kernel_reason_log import builtin_exception_name


# key path -> (key file fingerprint at load, or None for the fallback salt, secret).
_AUTH_BINDING_SECRET_CACHE: dict[str, tuple[tuple[int, ...] | None, bytes]] = {}
_AUTH_BINDING_FALLBACK_SECRET: bytes | None = None
_AUTH_BINDING_SECRET_LOCK = threading.RLock()


def _valid_secret_file(info: os.stat_result, *, uid: int) -> bool:
    return (
        stat.S_ISREG(info.st_mode)
        and stat.S_IMODE(info.st_mode) == 0o600
        and info.st_uid == uid
        and info.st_size == 32
    )


def _read_secret(path: Path, *, failure_class: list[str] | None = None) -> bytes | None:
    """Read only an owned, regular, exact-mode installation key."""
    descriptor = None
    try:
        uid = os.getuid()
        path_info = os.lstat(path)
        if not _valid_secret_file(path_info, uid=uid):
            return None
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(path, flags)
        info = os.fstat(descriptor)
        if (not _valid_secret_file(info, uid=uid)
                or (path_info.st_dev, path_info.st_ino) != (info.st_dev, info.st_ino)):
            return None
        chunks: list[bytes] = []
        remaining = 33
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        secret = b"".join(chunks)
        if len(secret) != 32 or not _valid_secret_file(os.fstat(descriptor), uid=uid):
            return None
        return secret
    except Exception as error:
        # Storage and validation failures safely cause a remeasurement.
        if failure_class is not None:
            failure_class.append(_exception_class_name(error))
        return None
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _exception_class_name(error: Exception) -> str:
    """Return only an allowlisted builtin exception name for a local diagnostic."""
    return builtin_exception_name(error)


def _create_secret(
    path: Path,
    *,
    failure_class: list[str] | None = None,
) -> bytes | None:
    """Publish a new installation key with no wider-mode visibility window."""
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        secret = secrets.token_bytes(32)
        temporary = path.with_name(
            f".{path.name}.{os.getpid()}.{secrets.token_hex(16)}.tmp"
        )
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(temporary, flags, 0o600)
        try:
            # os.open's mode is filtered by umask; fchmod makes it exact.
            os.fchmod(descriptor, 0o600)
            view = memoryview(secret)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("identity key write did not complete")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

        try:
            os.link(temporary, path)
        except FileExistsError:
            # Another process won first use. Remove our temp before reading it.
            pass
        os.unlink(temporary)
        temporary = None
        _fsync_directory(path.parent)
        return _read_secret(path, failure_class=failure_class)
    except Exception as error:
        if failure_class is not None:
            failure_class.append(_exception_class_name(error))
        return None
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass


def _fallback_secret() -> bytes:
    global _AUTH_BINDING_FALLBACK_SECRET
    with _AUTH_BINDING_SECRET_LOCK:
        if _AUTH_BINDING_FALLBACK_SECRET is None:
            _AUTH_BINDING_FALLBACK_SECRET = secrets.token_bytes(32)
        return _AUTH_BINDING_FALLBACK_SECRET


def _report_process_local_fallback(cause: str) -> None:
    """Report fallback without exposing identity paths, values, or exception messages."""
    try:
        if __package__:
            from .kernel_reason_log import emit_reason
        else:
            from kernel_reason_log import emit_reason
        emit_reason(
            "fallback", "installation_identity_process_local", "request_auth_identity",
            "using process-local installation identity",
            f"persistent identity key unavailable ({cause})", level="WARN", journal=False,
        )
    except Exception:
        # Diagnostics must never make request measurement unavailable.
        pass


def _register_checkpoint_secret(secret: bytes) -> None:
    # Import only the matching guard, then cover aliases that are already loaded.
    if __package__:
        from . import kernel_checkpoint as native_checkpoint
    else:
        import kernel_checkpoint as native_checkpoint
    native_checkpoint.register_secret(secret)
    for name in ("kernel_checkpoint", "engine.kernel_checkpoint", "stateframe.engine.kernel_checkpoint"):
        checkpoint = sys.modules.get(name)
        if checkpoint is not None and checkpoint is not native_checkpoint:
            checkpoint.register_secret(secret)


def _installation_home(configured_home: str | Path | None = None) -> Path:
    if __package__:
        from .kernel_safe_paths import safe_expanduser, safe_home
    else:
        from kernel_safe_paths import safe_expanduser, safe_home

    selected = configured_home if configured_home is not None else os.environ.get("STATEFRAME_HOME")
    if selected:
        home = safe_expanduser(Path(selected))
    else:
        user_home = safe_home()
        if user_home is None:
            raise ValueError("cannot find the StateFrame home: set HOME or STATEFRAME_HOME")
        home = safe_expanduser(user_home / ".stateframe")
    return home if home.is_absolute() else Path.cwd() / home


def _installation_secret(configured_home: str | Path | None = None) -> bytes:
    try:
        key_path = _installation_home(configured_home) / "identity.key"
        cache_key = os.fspath(key_path)
    except Exception as error:
        secret = _fallback_secret()
        _report_process_local_fallback(_exception_class_name(error))
        return secret

    creation_failure: list[str] = []
    read_failure: list[str] = []
    used_fallback = False
    with _AUTH_BINDING_SECRET_LOCK:
        cached = _AUTH_BINDING_SECRET_CACHE.get(cache_key)
        # A path is not an identity: a long-lived process can see the key at the same path
        # replaced, rotated or deleted (home recreated between runs). Reuse the cached secret
        # only while the file is the one it was loaded from; otherwise reload, so the secret
        # registered for checkpoint redaction is always the key actually on disk.
        if cached is not None and cached[0] == _key_fingerprint(key_path):
            _register_checkpoint_secret(cached[1])
            return cached[1]
        secret = _read_secret(key_path, failure_class=read_failure)
        if secret is None:
            secret = _create_secret(
                key_path,
                failure_class=creation_failure,
            )
        fingerprint = _key_fingerprint(key_path) if secret is not None else None
        if (fingerprint is not None
                and _read_secret(key_path, failure_class=read_failure) != secret):
            fingerprint = None  # replaced while loading: never matches a valid file, so reload
        if secret is None:
            used_fallback = True
            secret = _fallback_secret()
        _AUTH_BINDING_SECRET_CACHE[cache_key] = (fingerprint, secret)
        _register_checkpoint_secret(secret)
    if used_fallback:
        read_cause = read_failure[-1] if read_failure else None
        if read_cause == "FileNotFoundError":
            read_cause = None  # Missing key is the normal first-use path.
        cause = read_cause or (
            creation_failure[-1] if creation_failure else "IdentityKeyUnavailable"
        )
        _report_process_local_fallback(cause)
    return secret


def _key_fingerprint(path: Path) -> tuple[int, ...] | None:
    """Identity of the key file as loaded; None when there is no valid file to identify."""
    try:
        info = os.lstat(path)
    except OSError:
        return None
    if not _valid_secret_file(info, uid=os.getuid()):
        return None
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def protected_home_identity(
        configured_home: str | Path | None = None, *, initialize: bool = False,
) -> dict[str, int | str]:
    """Return the filesystem identity of the protected home and its identity key.

    Initialization is runner-start-only. Snapshot and export checks never recreate a
    missing home or key, because a disappearance means the protected identity changed.
    """
    home = Path(os.path.realpath(os.fspath(_installation_home(configured_home))))
    key_path = home / "identity.key"
    if initialize:
        _installation_secret(home)
    try:
        home_info = os.stat(home, follow_symlinks=False)
        key_info = os.lstat(key_path)
        secret = _read_secret(key_path)
        home_after = os.stat(home, follow_symlinks=False)
        key_after = os.lstat(key_path)
    except OSError as error:
        raise ValueError("protected home identity is unavailable") from error
    if secret is None:
        raise ValueError("protected identity key is unavailable")
    if (not stat.S_ISDIR(home_info.st_mode) or not stat.S_ISREG(key_info.st_mode)
            or (home_info.st_dev, home_info.st_ino) != (home_after.st_dev, home_after.st_ino)
            or not stat.S_ISDIR(home_after.st_mode)
            or any(getattr(key_info, name) != getattr(key_after, name) for name in (
                "st_dev", "st_ino", "st_mode", "st_uid", "st_size", "st_mtime_ns", "st_ctime_ns"))):
        raise ValueError("protected home or key changed during identity capture")
    _register_checkpoint_secret(secret)
    return {
        "realpath": os.fspath(home),
        "home_st_dev": int(home_info.st_dev),
        "home_st_ino": int(home_info.st_ino),
        "key_st_dev": int(key_info.st_dev),
        "key_st_ino": int(key_info.st_ino),
        "key_st_ctime_ns": int(key_info.st_ctime_ns),
    }


def auth_binding_for_key(key: str | None) -> str | None:
    """Return HMAC-SHA256 of the auth key under this installation's secret."""
    if key is None:
        return None
    return hmac.new(
        _installation_secret(), key.encode("utf-8", errors="surrogatepass"), hashlib.sha256,
    ).hexdigest()
