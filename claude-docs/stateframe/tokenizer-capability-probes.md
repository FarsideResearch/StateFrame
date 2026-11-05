# StateFrame provider capability probes

`kernel_provider_dialect.capabilities_for` classifies a provider per origin and stores the result in the run capability cache. A valid llama.cpp `/props` response proves the dialect. It does not prove that `/slots` is available or that any slot ID is usable.

Request tokenization needs the dialect proof but has a bounded request deadline, so it may skip `/slots`. That result is cached with `slots=False` and `slots_status=None`; `None` means slot status is still unproven. A later caller that needs slots probes `/slots` and upgrades the cached record while retaining the dialect already proven by `/props`. A timed-out slot probe leaves that proof intact, while any feature that requires slots still needs a successful slot proof.

The tokenizer regression coverage is in `tests/test_request_tokenizer_reprobe.py`. The clean-meter wire oracle is pinned to the 012e0b5 fixture, whose tape server had no `/props` route; its capture opts out of `/props` so the request-byte comparison keeps the same provider-probe contract as the oracle.
