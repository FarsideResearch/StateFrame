# StateFrame container

No image is pushed or published by this project: build it yourself. The image contains Debian packages under the GPL and other licenses; if you redistribute an image, you must meet their terms.

The recipe builds the release recorded in `pyproject.toml` at the selected clean commit ref. It uses the official Python 3.13 slim Bookworm image. Resolve and review its digest before building:

```sh
PYTHON_IMAGE="$(docker/resolve-digest.sh python:3.13-slim-bookworm)" BUILDER="docker buildx" docker/build.sh <clean-commit-ref>
```

`docker/build.sh` requires BuildKit through Docker Buildx. `BUILDER` can name another BuildKit-capable command prefix. Optional `BUILD_CPUS` accepts a whole CPU count from 1 through 256; `BUILD_MEMORY` accepts a positive byte value such as `2g` or `512m`. The output image tag uses the version from the selected ref. The build installs the hash-locked runtime requirements, checks wheel digest in both stages, and validates package metadata and required license files in the first stage.

Set `STATEFRAME_IMAGE` to the exact versioned tag printed by `docker/build.sh` before running `docker/smoke/smoke.sh`; the smoke script requires this value so its image always matches the selected build ref.

The image contains no endpoint credentials. Set `STATEFRAME_ENDPOINT`, `STATEFRAME_MODEL`, and any API key environment variable when the container starts. Pass the key variable name with `--api-key-env`; do not pass the key value in arguments. The Kubernetes example reads the key from a Secret.

`STATEFRAME_ALLOW_REMOTE_ENDPOINT=1` allows the runner to send repository content to any non-loopback HTTP or HTTPS endpoint and to follow non-loopback redirects.

For smoke runs, `SMOKE_EXTRA_ARGS` accepts additional whitespace-separated `stateframe run` arguments. The Kubernetes Job reads the same kind of optional argument list from `STATEFRAME_EXTRA_ARGS`; it adds `--api-key-env` only when the optional Secret key is present.

The entrypoint runs under `tini`, which forwards signals and reaps orphaned child processes in Docker and Kubernetes. `/home/sf` and `/workspace` are group 0 writable for arbitrary numeric UIDs. The image declares no Docker volumes, so `--read-only` does not create implicit writable anonymous volumes. For a run, mount writable storage at `/home/sf` and `/workspace` and provide a writable `/tmp`; `--version` can run with only a read-only root filesystem.

The Docker smoke script does not enable container mode. Its startup probe requires bubblewrap and Linux user namespaces; on this setup Docker's default seccomp denies the user namespace probe, so plain smoke runs return the typed `REFUSED SANDBOX_UNAVAILABLE` failure. The smoke script reports this as `FAIL`. Run it only where the probe succeeds without weakening container or host security settings.

The Kubernetes Job reads runtime values from a ConfigMap and an optional key from a Secret. It defaults to `--container-is-sandbox --shell unrestricted --shell-network full` because pods do not provide bubblewrap. Use this mode with trusted repositories only: code in the pod can read the provider key and reach whatever the NetworkPolicy allows. See the [docs/guide/container.md](../docs/guide/container.md) for the container boundary and resume behavior.

Replace the Job's `sf-eks-stateframe:replace-with-build-tag` image with the exact versioned tag printed by `docker/build.sh` before applying the manifests. Preload the `sf-eks-repository` claim at `/workspace/repo`. The NetworkPolicy permits cluster DNS and the configured endpoint IP and port. Replace the reserved example endpoint range and port in `docker/k8s/base/networkpolicy.yaml`.

The Job's default run directory is under `/home/sf`, mounted through the `home` volume. The base render uses `emptyDir` for `home` and `workspace`. Use the PVC overlay to keep both on volume claims, so the run directory outlives the pod. The Job only starts new runs (`stateframe run`, no retries); to resume, start a pod with the same claims and run `stateframe resume RUN_DIR --container-is-sandbox`. That has not been run on a cluster.
