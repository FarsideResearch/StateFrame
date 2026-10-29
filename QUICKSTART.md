# Quickstart

You need macOS on Apple silicon (with the built-in `sandbox-exec`) or Linux x86_64 (tested on Ubuntu 22.04; needs bubblewrap: `sudo apt install bubblewrap`), Python 3.11 or newer, free disk of a few GB per long run (the run directory holds the history), and a local OpenAI-compatible server with tool calling that is already running. For llama.cpp, each slot needs 65,536 tokens of context: `-c` is the total, so use `-c 65536` times your `--parallel` slot count (StateFrame refuses a server whose slots are smaller). With `--kv-unified`, `-c` is the TOTAL cache shared by all slots, and every slot reports that total, so StateFrame cannot catch an undersized unified cache: still use `-c 65536` times `--parallel` (for example `-np 4 -c 262144 --kv-unified`). Add `--ctx-checkpoints 32 -cms 256` plus a `-cram` cache size that fits your machine's RAM.

Your tests run with your project's Python, not StateFrame's: the repository's `.venv/`, `venv/` or `env/` if there is one, else `python3` on PATH. That interpreter needs your project's dependencies and pytest. Override it with `--test-python PATH`; StateFrame prints the one it chose at the start of each run.

Install, then apply the PATH hint the installer prints:

```bash
./install.sh
```

Point StateFrame at your server, then run these from the unpacked package directory (it holds `examples/`):

```bash
export STATEFRAME_ENDPOINT=http://127.0.0.1:8080/v1
export STATEFRAME_MODEL=my-local-model
stateframe run --repo examples/discount_bug --task "Discounts are charged wrong at checkout: a 25% discount on an 80.00 order should cost 60.00, and a 100% discount should be free. Fix the pricing code so the tests pass."
stateframe run --repo examples/tiny_site --task "Build a small static shop site in site/: index.html with a header, a list of 3 product cards (name, price, button) and a footer; styles.css linked from index.html with a simple card layout; app.js linked from index.html that adds a click handler to each button showing the product name; plus products.json holding the 3 products, with app.js rendering the cards from it. Make tests/test_site.py pass."
stateframe inspect <run-dir> --send 2 --frame
stateframe chat --repo path/to/your-repo
```

Each run prints its run directory; pass it to `stateframe inspect`. There is no time limit: stop any run with Ctrl-C.

## Model server on another machine: co-locate the tokenizer

The engine counts tokens many times per turn, by default on the model server. If that server is remote,
run a second llama.cpp server with the same model file on this machine (it only tokenizes: no GPU, small
context) and point the engine at it:

```bash
llama-server -m <same-model>.gguf --port 8081 -c 2048 -ngl 0
export LLAMA_CPP_TOKENIZE_URL=http://127.0.0.1:8081
```

Check it is the same tokenizer before a run. The two token id lists must be identical:

```bash
for u in http://127.0.0.1:8081 http://<model-server>:8080; do
  curl -s "$u/tokenize" -H 'Content-Type: application/json' -d '{"content":"def add(a, b):\n    return a + b  # héllo"}'; echo
done
```

Different ids mean a different tokenizer: wrong counts can overflow the context or trim too much.

## License

StateFrame is licensed under the GNU Affero General Public License v3.0 only (AGPL-3.0-only), or under a separate commercial license from Farside Research LLC (see COMMERCIAL-LICENSE.md). Third-party dependencies keep their own licenses; see THIRD_PARTY_LICENSES.md.
