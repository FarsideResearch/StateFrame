# syntax=docker/dockerfile:1.4
# Builds must pass a resolved digest. This valid tag keeps FROM checks clean while the
# wheel verifier rejects an unpinned image when the build argument is omitted.
ARG PYTHON_IMAGE=python:3.13-slim-bookworm
FROM ${PYTHON_IMAGE} AS wheel-check

ARG PYTHON_IMAGE
ARG APP_VERSION
ARG WHEEL_SHA256
COPY dist/*.whl /tmp/wheels/
COPY docker/verify_wheel.py /tmp/verify_wheel.py

RUN python /tmp/verify_wheel.py /tmp/wheels \
    --python-image "$PYTHON_IMAGE" --app-version "$APP_VERSION" \
    --sha256 "$WHEEL_SHA256" --licenses-dir /usr/share/licenses/stateframe

FROM ${PYTHON_IMAGE} AS runtime

ARG PYTHON_IMAGE
ARG APP_VERSION
ARG WHEEL_SHA256

LABEL org.opencontainers.image.title="StateFrame" \
      org.opencontainers.image.version="${APP_VERSION}" \
      org.opencontainers.image.description="Open Agent Engine for local chat completion servers" \
      org.opencontainers.image.licenses="AGPL-3.0-only"

ENV HOME=/home/sf \
    PATH=/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates git bubblewrap tini \
 && rm -rf /var/lib/apt/lists/* \
 && install -d -o root -g 0 -m 2775 /home/sf /workspace

COPY --from=wheel-check /tmp/wheels/ /tmp/wheels/
COPY --from=wheel-check /usr/share/licenses/stateframe/ /usr/share/licenses/stateframe/
COPY requirements.txt /tmp/requirements.txt

RUN python - "${PYTHON_IMAGE}" "${WHEEL_SHA256}" <<'PY'
import hashlib
import re
import sys
from pathlib import Path

base_name = sys.argv[1].rsplit("/", 1)[-1]
if not re.fullmatch(r"python:3\.13-slim-bookworm@sha256:[0-9a-f]{64}", base_name):
    raise SystemExit("PYTHON_IMAGE must be Python 3.13 slim Bookworm pinned by sha256")
expected_sha = sys.argv[2]
if not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
    raise SystemExit("WHEEL_SHA256 must be 64 lowercase hex digits")
wheels = list(Path("/tmp/wheels").glob("*.whl"))
if len(wheels) != 1:
    raise SystemExit(f"expected exactly one wheel, found {len(wheels)}")
digest = hashlib.sha256()
with wheels[0].open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
if digest.hexdigest() != expected_sha:
    raise SystemExit("wheel SHA-256 does not match WHEEL_SHA256")
PY
RUN python -m pip install --no-cache-dir --require-hashes --only-binary=:all: -r /tmp/requirements.txt \
 && python -m pip install --no-cache-dir --no-deps /tmp/wheels/*.whl \
 && rm -rf /tmp/wheels /tmp/requirements.txt /root/.cache/pip \
 && chmod -R a=rX /usr/share/licenses/stateframe

WORKDIR /workspace
USER 65532:0
STOPSIGNAL SIGTERM
ENTRYPOINT ["/usr/bin/tini", "-g", "--", "stateframe"]
CMD ["--help"]
