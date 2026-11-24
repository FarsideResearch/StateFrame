"""Differential and work-count checks for streaming secret matching."""
from __future__ import annotations

import builtins
import random

import kernel_secret_scan
from kernel_secret_scan import SecretScanner, _JsonStringUnescaper


class _BaseSecretScanner:
    """The 0.8.30 scanner logic, retained as a plain behavioral reference."""

    def __init__(self, secrets: list[bytes], *, stages: int = 8) -> None:
        self.secrets = tuple(secrets)
        self.hex_secrets = tuple(set(secrets) & {value.hex().encode("ascii") for value in secrets})
        self.keep = max((len(value) for value in secrets), default=1) - 1
        self.chunk_size = max(65536, self.keep + 1)
        self.decoders = [_JsonStringUnescaper() for _ in range(stages)]
        self.tails = [b""] * (stages + 1)
        self.found = False
        self.finished = False

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
        if not self.finished and not self.found and self.secrets:
            decoded = b""
            for index, decoder in enumerate(self.decoders, 1):
                decoded = decoder.feed(decoded, final=True)
                if self._match(decoded, index):
                    break
        self.finished = True
        return self.found


def _state(scanner):
    return (scanner.found, scanner.finished, tuple(scanner.tails),
            tuple(decoder.pending for decoder in scanner.decoders))


def _generator_steps(monkeypatch, operation) -> int:
    original_any = builtins.any
    steps = 0

    def counted_any(values):
        def counted_values():
            nonlocal steps
            for value in values:
                steps += 1
                yield value

        return original_any(counted_values())

    monkeypatch.setattr(builtins, "any", counted_any)
    try:
        operation()
    finally:
        monkeypatch.setattr(builtins, "any", original_any)
    return steps


def test_repeated_match_work_drops_for_20000_chunks(monkeypatch):
    secrets = [f"credential-{index:03d}".encode() for index in range(111)]
    payload = b"unrelated workspace content"
    reference = _BaseSecretScanner(secrets)
    optimized = SecretScanner(secrets)
    entries = 20_000

    def scan(scanner):
        for _ in range(entries):
            scanner.feed(payload)

    reference_steps = _generator_steps(
        monkeypatch, lambda: scan(reference))
    optimized_steps = _generator_steps(
        monkeypatch, lambda: scan(optimized))

    assert reference_steps == entries * len(secrets) * (1 + len(reference.decoders))
    assert optimized_steps == 0


def test_compiled_matches_equal_base_for_random_chunks_and_decoding():
    rng = random.Random(0xD7)
    cases = [
        ([b"needle", bytes.fromhex("deadbeef"), b"deadbeef"],
         b'{"raw":"needle","escaped":"nee\\u0064le","hex":"DEADBEEF"}'),
        ([b"", b"\\u", b"overlap", b"rlap"], b'xxoverlap\\u0061'),
    ]
    for _ in range(100):
        secrets = [rng.randbytes(rng.randint(1, 9)) for _ in range(rng.randint(1, 8))]
        if rng.randrange(3) == 0:
            secrets.append(b"")
        data = rng.randbytes(rng.randint(0, 100))
        if secrets and secrets[0]:
            offset = rng.randrange(len(data) + 1)
            data = data[:offset] + secrets[0] + data[offset:]
        cases.append((secrets, data))

    for case_index, (secrets, data) in enumerate(cases):
        reference = _BaseSecretScanner(secrets)
        optimized = SecretScanner(secrets)
        chunk_size = 1 if case_index < 2 else rng.randint(1, 17)
        reference.chunk_size = optimized.chunk_size = chunk_size
        offset = 0
        while offset < len(data):
            width = 1 if case_index == 0 else rng.randint(1, 13)
            end = min(len(data), offset + width)
            part = data[offset:end]
            assert optimized.feed(part) == reference.feed(part)
            assert _state(optimized) == _state(reference)
            offset = end
        assert optimized.finish() == reference.finish()
        assert _state(optimized) == _state(reference)


def test_matcher_input_snapshot_and_reassigned_secret_inputs_stay_fresh():
    caller_secrets = [b"original"]
    scanner = SecretScanner(caller_secrets)
    caller_secrets.append(b"outside-mutation")
    assert not scanner._match(b"outside-mutation", 0)

    scanner.secrets = [b"first"]
    assert scanner._match(b"first", 0)
    scanner.secrets.append(b"later")
    assert scanner._match(b"later", 0)

    scanner = SecretScanner([b"other"])
    scanner.hex_secrets = [b"needle"]
    assert scanner._match(b"NEEDLE", 0)


def test_unavailable_compiled_pattern_falls_back_to_base_matching(monkeypatch):
    monkeypatch.setattr(kernel_secret_scan, "_literal_pattern", lambda *_args, **_kwargs: None)
    scanner = SecretScanner([b"needle"])

    assert scanner._match(b"prefix-needle-suffix", 0)
