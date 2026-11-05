## 2026-11-04

### StateFrame — tokenizer capability proof caching
- Keep successful `/props` llama.cpp proof in the run cache when tokenization skips `/slots`; later slot-required calls probe and record slot status separately.
- Make the clean-meter capture use the pre-`/props` tape contract recorded by the 012e0b5 wire oracle, keeping the byte comparison on its original input contract.
- Files touched: `kernel_provider_dialect.py`, `tests/test_request_tokenizer_reprobe.py`, `tests/test_allocation_replay_corpus.py`, `tests/tape_server.py`, `claude-docs/stateframe/tokenizer-capability-probes.md`.
