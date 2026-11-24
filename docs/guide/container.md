# Running inside a container

`--container-is-sandbox` is opt in and is for trusted repositories only. It is off by default and treats the current container as the outer boundary. The engine does not create a private filesystem, network, or PID namespace. Code run for the task may read the provider key, reach the network, and alter run records.

StateFrame must detect that it is inside a container, and the process must run as a non-root user.

## Shell levels

`--shell standard` (the default), `--shell unrestricted` and `--shell off` are accepted. The standard program filter is a convenience applied to the command text; with the command sandbox gone it does not limit what an admitted interpreter or build script can do. Container mode refuses `--shell read-only` and unknown shell levels with `CONTAINER_CONFLICT` because the container cannot enforce those restrictions. For `run`, also pass `--shell-network full` explicitly. The default network setting does not qualify.

## Refused options

Container mode refuses `--shell-sandbox off`, `--yolo`, mutation runs, MCP, and parallel children. Hooks are refused with `CONTAINER_UNSUPPORTED:hooks`. An attempt to use the optional Python probe is refused with `CONTAINER_UNSUPPORTED:python-probe`.

## Settings that only the sandbox enforces

These settings are normally enforced by StateFrame's command sandbox, which container mode does not use:

| Setting | Behavior in container mode |
|---|---|
| Edit scope | Workspace changes are diffed and out-of-scope workspace changes are restored. Files outside the workspace are not covered. |
| Protected tests | Protected tests are checked by hash after commands. Changes are detected after a command, not prevented. |
| Shell read roots | They do not restrict reads in container mode. |
| Environment and secrets | The command environment is filtered, but readable files, sockets, and `/proc` entries for the engine are not hidden. |
| Process limits | Process limits still apply, without a per-command cgroup in this setup. |

## Resume

Pass `--container-is-sandbox` again when resuming, for example `stateframe resume RUN_DIR --container-is-sandbox`. The saved run must record `--shell-network full` and a shell level of `standard`, `unrestricted` or `off`. A missing shell value is refused with `CONTAINER_CONFLICT`. A resume in a different container compares only the saved and current backend, policy, and container classification.

If a shell command is killed while it is running, its result may be unknown. Resume refuses and asks you to reconcile the operation and start a new run.

## Docker example

No image is published. Build it yourself with `docker/build.sh`. The base image contains Debian packages under the GPL and other licenses; redistributing an image you build means meeting those terms.

Set `STATEFRAME_ENDPOINT`, `STATEFRAME_MODEL`, and `API_KEY` in your shell first. The endpoint must be reachable from the container. Replace `stateframe-image:TAG` with the image you built. Make the host run directory writable by UID 54321 or group 0. The run directory must be on a mounted directory to be resumable.

```sh
docker run --rm \
  --user 54321:0 \
  --read-only \
  --cap-drop ALL \
  --security-opt no-new-privileges:true \
  --pids-limit 1024 \
  --cpus 4 \
  --memory 8g \
  --tmpfs /tmp:mode=1777,exec \
  --tmpfs /home/sf:mode=1777,exec \
  --volume "$PWD:/repo:ro" \
  --volume "$HOME/stateframe-runs:/runs" \
  -e STATEFRAME_ENDPOINT \
  -e STATEFRAME_MODEL \
  -e API_KEY \
  stateframe-image:TAG \
  run --repo /repo \
  --run-dir /runs/discount-bug-1 \
  --task "Fix the issue and run the tests" \
  --api-key-env API_KEY \
  --container-is-sandbox \
  --shell unrestricted \
  --shell-network full
```

Use a new path under `/runs` for each run. To resume, start another container with the same `/runs` mount and run `stateframe resume /runs/discount-bug-1 --container-is-sandbox`. The default run location is under `/home/sf`, which is a temporary filesystem in this example.

If `STATEFRAME_ENDPOINT` is non-loopback from inside the container, add `-e STATEFRAME_ALLOW_REMOTE_ENDPOINT=1` to `docker run`. Endpoint validation requires this explicit opt in for a remote URL and warns that source code will be sent there. A routed endpoint can instead use that route's `allow_remote = true` setting. These options permit the endpoint; they do not make it reachable. Do not add the environment override for loopback endpoints.

Only pass credentials and mount files that you are willing to expose to the repository's code. Task code in the container or pod can read the API key and can use any egress that is allowed. The Kubernetes NetworkPolicy does nothing unless the cluster's CNI enforces it.

## Limits

OpenTelemetry export is a separate command that you run on a finished run. It is not part of the container boundary, which covers the engine run only.

A scripted local endpoint was run in real Docker containers, locked down as in the example, including kill and resume from a mounted directory; not tested: Kubernetes, EKS, a real model endpoint, Podman, or kubelet signals. Landlock is not implemented.
