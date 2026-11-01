"""One resend of a refused gzip request using its original identity bytes."""
from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar
import urllib.error


_Response = TypeVar("_Response")


def send_with_gzip_fallback(
        send: Callable[[bytes, dict[str, str]], _Response],
        body: bytes, headers: dict[str, str], identity: bytes,
        identity_headers: dict[str, str],
        on_refused: Callable[[urllib.error.HTTPError], bool]) -> _Response:
    """Retry only the first HTTP 415 to a request actually encoded as gzip."""
    try:
        return send(body, headers)
    except urllib.error.HTTPError as error:
        if (error.code != 415 or headers.get("Content-Encoding") != "gzip"
                or not on_refused(error)):
            raise
    return send(identity, identity_headers)
