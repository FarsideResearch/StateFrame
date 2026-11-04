"""Meter evidence for default final cache layout and hosted endpoints."""
from __future__ import annotations

import json

from kernel_context import token_estimate
from kernel_request_diet import RequestTokenMeter
from stateframe.scripted_endpoint import Script, serve

from test_final_cache_layout_host import _scripted_wire


EXAMPLES = (
    ("discount_bug", "Fix the pricing code so the tests pass."),
    ("tiny_site", "Build the small static shop site described in the README."),
    ("brief_to_report", "Write the report the brief asks for."),
)


def test_estimate_overcounts_every_bundled_scripted_request(tmp_path):
    bodies = []
    for example, task in EXAMPLES:
        label = f"meter-{example}"
        (tmp_path / f"launch-{label}").mkdir()
        wires = _scripted_wire(tmp_path, label, None, example=example, task=task)
        bodies.extend(json.loads(wire) for wire in wires)

    tokenize_script = Script({"model": "scripted", "steps": [], "tokenize": {"mode": "ok"}})
    server = serve(tokenize_script)
    margins: list[tuple[int, float]] = []
    try:
        endpoint = f"http://127.0.0.1:{server.server_port}/v1/tokenize"
        meter = RequestTokenMeter(endpoint=endpoint, route_allow_remote=True,
                                  template_overhead_tokens=0)
        for body in bodies:
            measured, method = meter.measure(body)
            estimate = token_estimate(body)
            assert method == "tokenize"
            margin = estimate - measured
            percent = margin * 100 / max(measured, 1)
            margins.append((margin, percent))
            assert margin >= 0, f"estimate under-counted by {-margin} tokens"
    finally:
        server.shutdown()
        server.server_close()

    minimum_tokens, minimum_percent = min(margins, key=lambda item: item[0])
    print(f"minimum conservative-estimate margin: {minimum_tokens} tokens ({minimum_percent:.2f}%) "
          f"across {len(bodies)} requests; scripted /tokenize is a len(content)/4 test meter")
    assert bodies
