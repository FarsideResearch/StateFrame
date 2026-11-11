"""Large, valid JSON escape fixtures shared by credential export regressions."""
from functools import lru_cache


@lru_cache(maxsize=4)
def seven_layer_json(key: bytes) -> bytes:
    """Encode every byte at every layer, expanding a 32-byte key beyond 9 MB."""
    escapes = tuple(f"\\u{byte:04x}".encode("ascii") for byte in range(256))
    payload = key
    for _ in range(7):
        payload = b'"' + b"".join(escapes[byte] for byte in payload) + b'"'
    return payload
