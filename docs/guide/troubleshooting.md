# Troubleshooting

| Message (or symptom) | Meaning | Fix |
|---|---|---|
| `test interpreter ... cannot import pytest` | The Python chosen for your tests has no pytest. | Install pytest in your project's environment, or pass `--test-python` with one that has it. |
| `no interpreter for your tests` | No `.venv/`, `venv/` or `env/` in the repository, and no usable `python3` on PATH. | Pass `--test-python PATH`. |
| Before-checks fail with import errors | The test Python lacks your project's dependencies. | Point `--test-python` at your project's environment. |
| `endpoint does not serve model ...` | The server lists other model names. | Pass `--model` with a name it lists. |
| `STATEFRAME_CONTEXT_TOKENS=N requires a live server slot of N tokens; observed M` | The server's slot size differs from the configured context size (they must be equal; `--kv-unified` reports the total). | Start the server with a slot of exactly N (`-c` = N x `--parallel`, without `--kv-unified`), or unset the setting. See [Context size](running.md#context-size). |
| Slot too small / refused server | Each server slot needs 65,536 tokens (the default; see Context size in running.md). | For llama.cpp use `-c 65536` times your `--parallel` count. |
| `Context size has been exceeded` from the server when runs overlap | With `--kv-unified` all slots share one cache, and each slot reports the total, so the slot check passes while concurrent runs overflow it. | Start the server with `-c 65536` times `--parallel` (for example `-np 4 -c 262144 --kv-unified`). |
| Remote endpoint refused | Only loopback endpoints are allowed by default. | Set `STATEFRAME_ALLOW_REMOTE_ENDPOINT=1` if you accept that code leaves the machine. |
| Stops asking for `--target` / `--edit` | The repository has more than 200 source files. | Name the starting file and the other files the change may touch. |
| `unconsumed STATEFRAME settings: ...` (HARNESS_CONTRACT) | A profile sets a switch this engine does not read. | Remove it or fix its name. |
| Stops with a refusal about the prompt budget | The task state no longer fits 33,344 tokens (very long tasks, changes of around 10+ files). | Split the task; the partial result is saved. |
| Run ends with `STALLED` after repeated unchanged reads | A completed batch made entirely of unchanged reads costs one no-progress step, but repeated batches still reach the limit. | Inspect the last frame with `stateframe inspect <run-dir> --send N --frame`. Check whether the relevant source is visible and whether the current failing check gives a usable `file:line`; setup and collection failures, and failures without a source location, do not activate code bookmarks. See [Multi-file source and repeated reads](running.md#multi-file-source-and-repeated-reads). |
| `ERR:FrameFit5ImpossibleMinimum`, or `token counting unavailable: using conservative estimate` | The server has no `/tokenize` endpoint (for example Ollama or hosted APIs), so the engine estimates token counts conservatively, and even the minimum frame looks too big. | Use llama.cpp, or run a local llama.cpp tokenizer and set `LLAMA_CPP_TOKENIZE_URL`. A scripted endpoint needs `"tokenize": {"mode": "ok"}`. |
| Checks report FAIL with exit 71 and `sandbox-exec: execvp() ... Operation not permitted` | The checks could not start: the sandbox could not launch the test Python. We have seen it when StateFrame was started from a process that was itself sandboxed. It is not a test failure. | Run StateFrame from a normal terminal session. |
| `bwrap is required on Linux for StateFrame's command isolation (install it: sudo apt install bubblewrap)` (installer) | Linux needs bubblewrap for the sandbox. | `sudo apt install bubblewrap`, then run `./install.sh` again. |
| `test interpreter /usr/bin/python3 ... cannot import pytest` on Ubuntu | The stock `python3` (3.10 on Ubuntu 22.04) has no pytest and is older than StateFrame needs. | Install Python 3.11 or newer for StateFrame (`sudo apt install python3.11 python3.11-venv`), and pass `--test-python` with your project's environment. |
| Long waits per turn with a remote server | Token counting goes over the network. | Run a local tokenizer and set `LLAMA_CPP_TOKENIZE_URL` (QUICKSTART). |
| Requests rejected for context overflow after a tokenizer outage | The fallback estimate undercounted. | Keep the tokenizer up for the whole run. |
| `stateframe resume` refuses | The run already finished, its records are damaged, or a different engine version is installed. | The message names which; a different engine needs the original install. |
| `resume-divergence.json` appears | A rebuilt request differed from its record; the run continued live from there. | Nothing to do; the file says where. |
