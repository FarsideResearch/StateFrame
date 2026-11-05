"""Private, installation-scoped authentication bindings for request measurements."""
from __future__ import annotations

import hashlib
import hmac
import os
from pathlib import Path
import secrets
import stat
import threading


_AUTH_BINDING_SECRET_CACHE: dict[str, bytes] = {}
_AUTH_BINDING_FALLBACK_SECRET: bytes | None = None
_AUTH_BINDING_SECRET_LOCK = threading.RLock()


def _valid_secret_file(info: os.stat_result, *, uid: int) -> bool:
    return (
        stat.S_ISREG(info.st_mode)
        and stat.S_IMODE(info.st_mode) == 0o600
        and info.st_uid == uid
        and info.st_size == 32
    )


def _read_secret(path: Path) -> bytes | None:
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
    except Exception:
        # Storage and validation failures safely cause a remeasurement.
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


def _create_secret(path: Path) -> bytes | None:
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
        return _read_secret(path)
    except Exception:
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


def _installation_secret() -> bytes:
    try:
        # Reuse StateFrame's home resolution, including STATEFRAME_HOME and
        # the standard ~/.stateframe default.
        from stateframe.storage_gc import stateframe_home

        key_path = stateframe_home() / "identity.key"
        cache_key = os.fspath(key_path)
    except Exception:
        return _fallback_secret()

    with _AUTH_BINDING_SECRET_LOCK:
        cached = _AUTH_BINDING_SECRET_CACHE.get(cache_key)
        if cached is not None:
            return cached
        secret = _read_secret(key_path)
        if secret is None:
            secret = _create_secret(key_path)
        if secret is None:
            secret = _fallback_secret()
        _AUTH_BINDING_SECRET_CACHE[cache_key] = secret
        return secret


def auth_binding_for_key(key: str | None) -> str | None:
    """Return HMAC-SHA256 of the auth key under this installation's secret."""
    if key is None:
        return None
    return hmac.new(
        _installation_secret(), key.encode("utf-8", errors="surrogatepass"), hashlib.sha256,
    ).hexdigest()
