#!/bin/sh
# Export an existing finished run to the configured OTLP/HTTP receiver.
set -eu

if [ "$#" -ne 1 ]; then
    printf '%s\n' 'usage: scripts/otel_demo.sh RUN_DIR' >&2
    exit 2
fi

stateframe-otel export "$1" --trust-source
stateframe-otel status "$1" --json
