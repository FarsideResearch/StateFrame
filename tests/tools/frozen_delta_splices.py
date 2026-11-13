"""Offline splice experiment and dependency-free deterministic byte BPE verifier.

Run directly for the pinned BPE. For an offline llama.cpp vocabulary, pass
--llama-root (a built source tree) and --vocab (a GGUF file). A temporary helper
loads vocabulary only; no inference, GPU allocation or network is involved.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import random
import re
import subprocess
import tempfile
from pathlib import Path

TEXTS = (
    'def admission_tokens(body, meter):\n    return frozen.tokens + delta\n',
    '{"messages":[{"role":"user","content":"Task: fix tokenizer outage"}]}',
    '{"state":{"framefit5":{"next_site":{"path":"src/kernel.py"}}}}',
    '[Current source: pkg/a.py]\nrevision=sha256:123\nreturn count + 1',
    'internationalization counterexample unnecessarily conservative replacement',
    'résumé naïve café 中文 日本語 🙂🚀 e\u0301\u200d',
    '\\"\\n\\t null true false 1234567890 [] {},: _availability',
    'x' * 128,
)


class DeterministicBPE:
    """Ranked adjacent-byte merges, with fixed word/punctuation pre-tokenization."""

    def __init__(self):
        self.ranks = {bytes([value]): value for value in range(256)}
        self._counts = {}
        for text in TEXTS:
            for piece in self._pieces(text):
                encoded = piece.encode()
                for end in range(2, len(encoded) + 1):
                    self.ranks.setdefault(encoded[:end], len(self.ranks))

    @staticmethod
    def _pieces(text):
        return re.findall(r"\s+|[^\W\d_]+|\d+|[^\w\s]+|_", text)

    def _count_piece(self, piece):
        if piece in self._counts:
            return self._counts[piece]
        parts = [bytes([value]) for value in piece.encode()]
        while len(parts) > 1:
            choices = [(self.ranks[pair], index) for index in range(len(parts) - 1)
                       if (pair := parts[index] + parts[index + 1]) in self.ranks]
            if not choices:
                break
            _, index = min(choices)
            parts[index:index + 2] = [parts[index] + parts[index + 1]]
        if len(self._counts) < 8192:
            self._counts[piece] = len(parts)
        return len(parts)

    def __call__(self, text):
        return sum(self._count_piece(piece) for piece in self._pieces(text))


def measure(count, *, samples=20000, seed=12059):
    """Charge the entire excess to one seam, even for two-ended replacements."""
    rng = random.Random(seed)
    maximum = 0
    witness = None
    for _ in range(samples):
        old = rng.choice(TEXTS) + rng.choice(TEXTS)
        start = rng.randrange(len(old) + 1)
        end = rng.randrange(start, len(old) + 1)
        addition = rng.choice(TEXTS)
        left = rng.randrange(len(addition) + 1)
        addition = addition[left:rng.randrange(left, len(addition) + 1)]
        new = old[:start] + addition + old[end:]
        excess = count(new) - count(old) - count(addition)
        if excess > maximum:
            maximum = excess
            witness = [old, start, end, addition]
    return {"samples": samples, "seed": seed, "max_extra_tokens_per_splice": maximum,
            "witness": witness}


def llama_counter(root, vocabulary):
    """Compile against the local headers rather than guessing a ctypes ABI."""
    root = Path(root).resolve()
    source = '''#include "llama.h"
extern "C" void * sf_vocab(const char * path) {
    auto params = llama_model_default_params();
    params.vocab_only = true;
    params.n_gpu_layers = 0;
    auto model = llama_model_load_from_file(path, params);
    return model ? (void *) llama_model_get_vocab(model) : nullptr;
}
'''
    with tempfile.TemporaryDirectory(prefix="sf-delta-bpe-") as directory:
        helper = Path(directory) / "helper.cpp"
        library_path = Path(directory) / "helper.so"
        helper.write_text(source)
        subprocess.run(["c++", "-shared", "-fPIC", "-I", str(root / "include"),
                        "-I", str(root / "ggml/include"), str(helper),
                        "-L", str(root / "build/bin"),
                        "-Wl,-rpath," + str(root / "build/bin"), "-lllama",
                        "-o", str(library_path)], check=True)
        library = ctypes.CDLL(str(library_path))
    library.sf_vocab.argtypes = [ctypes.c_char_p]
    library.sf_vocab.restype = ctypes.c_void_p
    vocab = library.sf_vocab(str(vocabulary).encode())
    if not vocab:
        raise RuntimeError("vocabulary failed to load")
    library.llama_tokenize.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                     ctypes.c_int32, ctypes.c_void_p,
                                     ctypes.c_int32, ctypes.c_bool, ctypes.c_bool]
    library.llama_tokenize.restype = ctypes.c_int32

    def count(text):
        wire = text.encode()
        result = library.llama_tokenize(vocab, wire, len(wire), None, 0, False, False)
        return abs(result)

    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-root")
    parser.add_argument("--vocab")
    parser.add_argument("--samples", type=int, default=20000)
    args = parser.parse_args()
    count = DeterministicBPE()
    if args.llama_root:
        if not args.vocab:
            parser.error("--llama-root requires --vocab")
        count = llama_counter(args.llama_root, args.vocab)
    print(json.dumps(measure(count, samples=args.samples), ensure_ascii=False))


if __name__ == "__main__":
    main()
