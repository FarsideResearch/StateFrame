"""Bounded-memory credential matching through incremental JSON unescape stages.

These are lexical decoders, so JSON strings embedded in arbitrary containers are
covered too. Invalid escapes pass through literally; valid escapes use JSON's
UTF-8 representation, including surrogate pairs and lone surrogates.
"""
from __future__ import annotations

_SHORT_ESCAPES = {
    ord('"'): b'"', ord("\\"): b"\\", ord("/"): b"/", ord("b"): b"\b",
    ord("f"): b"\f", ord("n"): b"\n", ord("r"): b"\r", ord("t"): b"\t",
}
_HEX_DIGITS = b"0123456789abcdefABCDEF"


class _JsonStringUnescaper:
    """One incremental stage; pending holds at most eleven escape bytes."""

    def __init__(self) -> None:
        self.pending = b""

    def feed(self, data: bytes, *, final: bool = False) -> bytes:
        if not self.pending and b"\\" not in data:
            return data
        source = self.pending + data if self.pending else data
        self.pending = b""
        output = bytearray()
        position = 0
        size = len(source)
        while position < size:
            start = source.find(b"\\", position)
            if start < 0:
                output.extend(source[position:])
                break
            output.extend(source[position:start])
            if start + 1 == size:
                if final:
                    output.extend(source[start:])
                else:
                    self.pending = source[start:]
                break
            escape = source[start + 1]
            short = _SHORT_ESCAPES.get(escape)
            if short is not None:
                output.extend(short)
                position = start + 2
                continue
            if escape != ord("u"):
                output.append(ord("\\"))
                position = start + 1
                continue
            digits = source[start + 2:start + 6]
            if digits.strip(_HEX_DIGITS):
                output.append(ord("\\"))
                position = start + 1
                continue
            if len(digits) < 4:
                if final:
                    output.extend(source[start:])
                else:
                    self.pending = source[start:]
                break
            codepoint = int(digits, 16)
            position = start + 6
            if 0xD800 <= codepoint <= 0xDBFF:
                # Delay a high surrogate only while a low-surrogate escape can
                # still follow. The pair may straddle any input chunk boundary.
                following = source[position:position + 6]
                possible_pair = (b"\\u".startswith(following) if len(following) < 2
                                 else following.startswith(b"\\u")
                                 and not following[2:].strip(_HEX_DIGITS))
                if possible_pair and len(following) < 6 and not final:
                    self.pending = source[start:]
                    break
                if possible_pair and len(following) == 6:
                    low_digits = following[2:]
                    low = int(low_digits, 16)
                    if 0xDC00 <= low <= 0xDFFF:
                        codepoint = 0x10000 + ((codepoint - 0xD800) << 10) + low - 0xDC00
                        position += 6
            if codepoint < 0x80:
                output.append(codepoint)
            else:
                output.extend(chr(codepoint).encode("utf-8", errors="surrogatepass"))
        return bytes(output)


class SecretScanner:
    """Scan raw bytes and each of up to eight decoded streams in linear time.

    Every stream retains only the longest credential minus one byte for substring
    matching. Decoder state is independent of encoded key size. Work buffers are
    bounded by max(65536, key length), so total auxiliary space is O(stages * key
    length), including for callers that supply an entire multi-megabyte object.
    """

    def __init__(self, secrets: list[bytes], *, stages: int = 8) -> None:
        self.secrets = tuple(secrets)
        self.hex_secrets = tuple(set(secrets) & {value.hex().encode("ascii") for value in secrets})
        self.keep = max((len(value) for value in secrets), default=1) - 1
        self.chunk_size = max(65536, self.keep + 1)
        self.decoders = [_JsonStringUnescaper() for _ in range(stages)]
        self.tails = [b""] * (stages + 1)
        self.found = False
        self.finished = False

    @property
    def retained_bytes(self) -> int:
        return sum(map(len, self.tails)) + sum(len(stage.pending) for stage in self.decoders)

    def _match(self, data: bytes, stage: int) -> bool:
        if not data:
            return False
        text = self.tails[stage] + data
        self.found = any(value in text for value in self.secrets)
        if not self.found and self.hex_secrets:
            folded = text.lower()
            self.found = any(value in folded for value in self.hex_secrets)
        self.tails[stage] = text[-self.keep:] if self.keep else b""
        return self.found

    def feed(self, data: bytes) -> bool:
        if self.finished:
            raise ValueError("secret scanner is already finished")
        if self.found or not self.secrets:
            return self.found
        for start in range(0, len(data), self.chunk_size):
            decoded = data[start:start + self.chunk_size]
            if self._match(decoded, 0):
                return True
            for index, decoder in enumerate(self.decoders, 1):
                decoded = decoder.feed(decoded)
                if self._match(decoded, index):
                    return True
        return False

    def finish(self) -> bool:
        """Flush partial escapes at actual EOF, including a lone high surrogate."""
        if not self.finished and not self.found and self.secrets:
            decoded = b""
            for index, decoder in enumerate(self.decoders, 1):
                decoded = decoder.feed(decoded, final=True)
                if self._match(decoded, index):
                    break
        self.finished = True
        return self.found
