#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  printf 'usage: %s {run|sigterm|kill-resume}\n' "$0" >&2
}

if (( $# < 1 )); then usage; exit 2; fi
action=$1
shift
case "$action" in run|sigterm|kill-resume) ;; *) usage; exit 2 ;; esac
if (( $# != 0 )); then usage; exit 2; fi

command -v docker >/dev/null 2>&1 || { printf 'docker is required\n' >&2; exit 1; }
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
image=${STATEFRAME_IMAGE:-}
if [[ -z "$image" ]]; then
  printf 'STATEFRAME_IMAGE must be set to the exact versioned tag printed by docker/build.sh\n' >&2
  exit 2
fi
owner="sf-eks-smoke-$$-$RANDOM"
network="$owner-net"
work_volume="$owner-work"
home_volume="$owner-home"
server_name="$owner-server"
cli_name="$owner-run"
resume_name="$owner-resume"
test_name="$owner-test"
diff_name="$owner-diff"
seed_name="$owner-seed"
made_network=0
made_work=0
made_home=0
made_server=0
made_cli=0

container_owned() {
  local label_value
  label_value=$(docker container inspect --format '{{index .Config.Labels "sf-eks-smoke.owner"}}' "$1" 2>/dev/null) || return 1
  [[ "$label_value" == "$owner" ]]
}

volume_owned() {
  local label_value
  label_value=$(docker volume inspect --format '{{index .Labels "sf-eks-smoke.owner"}}' "$1" 2>/dev/null) || return 1
  [[ "$label_value" == "$owner" ]]
}

network_owned() {
  local label_value
  label_value=$(docker network inspect --format '{{index .Labels "sf-eks-smoke.owner"}}' "$1" 2>/dev/null) || return 1
  [[ "$label_value" == "$owner" ]]
}

cleanup() {
  if (( made_cli )) && container_owned "$cli_name"; then docker container rm -f "$cli_name" >/dev/null 2>&1 || true; fi
  if (( made_server )) && container_owned "$server_name"; then docker container rm -f "$server_name" >/dev/null 2>&1 || true; fi
  if (( made_work )) && volume_owned "$work_volume"; then docker volume rm "$work_volume" >/dev/null 2>&1 || true; fi
  if (( made_home )) && volume_owned "$home_volume"; then docker volume rm "$home_volume" >/dev/null 2>&1 || true; fi
  if (( made_network )) && network_owned "$network"; then docker network rm "$network" >/dev/null 2>&1 || true; fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for name in "$server_name" "$cli_name" "$resume_name" "$test_name" "$diff_name" "$seed_name"; do
  if docker container inspect "$name" >/dev/null 2>&1; then
    printf 'container already exists: %s\n' "$name" >&2
    exit 1
  fi
done
if docker network inspect "$network" >/dev/null 2>&1; then
  printf 'network already exists: %s\n' "$network" >&2
  exit 1
fi
for name in "$work_volume" "$home_volume"; do
  if docker volume inspect "$name" >/dev/null 2>&1; then
    printf 'volume already exists: %s\n' "$name" >&2
    exit 1
  fi
done

docker network create --internal --driver bridge --label "sf-eks-smoke.owner=$owner" "$network" >/dev/null
made_network=1
docker volume create --label "sf-eks-smoke.owner=$owner" "$work_volume" >/dev/null
if ! volume_owned "$work_volume"; then
  printf 'volume was not created by this smoke run: %s\n' "$work_volume" >&2
  exit 1
fi
made_work=1
docker volume create --label "sf-eks-smoke.owner=$owner" "$home_volume" >/dev/null
if ! volume_owned "$home_volume"; then
  printf 'volume was not created by this smoke run: %s\n' "$home_volume" >&2
  exit 1
fi
made_home=1

common=(
  --read-only
  --tmpfs /tmp
  --cap-drop ALL
  --security-opt no-new-privileges
  --user 54321:0
  --mount "type=volume,source=$work_volume,target=/workspace"
  --mount "type=volume,source=$home_volume,target=/home/sf"
)
docker run --rm --name "$seed_name" \
  --network none "${common[@]}" \
  --mount "type=bind,source=$script_dir/tiny_repo,target=/seed,readonly" \
  --entrypoint python "$image" -c \
  'import shutil; shutil.copytree("/seed", "/workspace/repo")'

server_args=(--host 0.0.0.0 --port 8765 --state-dir /workspace/.server-state)
if [[ "$action" == sigterm ]]; then server_args+=(--pause-at 1); fi
if [[ "$action" == kill-resume ]]; then server_args+=(--pause-at 2); fi
docker run --detach --name "$server_name" --label "sf-eks-smoke.owner=$owner" \
  --network "$network" --read-only --tmpfs /tmp --cap-drop ALL \
  --security-opt no-new-privileges --user 54321:0 \
  --mount "type=volume,source=$work_volume,target=/workspace" \
  --mount "type=volume,source=$home_volume,target=/home/sf" \
  --mount "type=bind,source=$script_dir/fake_server.py,target=/opt/fake_server.py,readonly" \
  --entrypoint python "$image" /opt/fake_server.py "${server_args[@]}" >/dev/null
made_server=1

ready=0
for _ in {1..100}; do
  if docker exec "$server_name" python -c \
    'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8765/v1/models", timeout=1).read()' \
    >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 0.2
done
if (( ! ready )); then
  docker logs "$server_name" >&2 || true
  printf 'fake endpoint did not become ready\n' >&2
  exit 1
fi

engine_flags=(--shell unrestricted --shell-sandbox on)
run_args=(
  run
  --repo /workspace/repo
  --task 'Read calc.py, change value() to return 42, run tests/test_calc.py, then finish.'
  --test-cmd 'python3 -m pytest -q tests/test_calc.py'
  --check tests/test_calc.py
  --in-place
  --run-dir /home/sf/smoke-run
  "${engine_flags[@]}"
)
smoke_extra_args=()
if [[ -n ${SMOKE_EXTRA_ARGS:-} ]]; then
  IFS=$' \t\n' read -r -d '' -a smoke_extra_args \
    < <(printf '%s\0' "$SMOKE_EXTRA_ARGS")
fi
run_args+=("${smoke_extra_args[@]}")
runtime_env=(
  -e "STATEFRAME_ENDPOINT=http://$server_name:8765/v1"
  -e STATEFRAME_MODEL=sf-eks-scripted
  -e STATEFRAME_ALLOW_REMOTE_ENDPOINT=1
)

report_typed_sandbox_failure() {
  if [[ "$1" == *'REFUSED SANDBOX_UNAVAILABLE'* ]]; then
    printf 'FAIL: StateFrame returned REFUSED SANDBOX_UNAVAILABLE; bubblewrap could not establish the required user namespace sandbox under the current container policy.\n' >&2
    return 0
  fi
  return 1
}

wait_for_pause() {
  local paused=0
  local cli_output
  for _ in {1..300}; do
    if docker exec "$server_name" python -c \
      'from pathlib import Path; raise SystemExit(0 if Path("/workspace/.server-state/paused").exists() else 1)' \
      >/dev/null 2>&1; then
      paused=1
      break
    fi
    if [[ $(docker container inspect --format '{{.State.Running}}' "$cli_name" 2>/dev/null || true) != true ]]; then
      break
    fi
    sleep 0.1
  done
  if (( ! paused )); then
    cli_output=$(docker logs "$cli_name" 2>&1 || true)
    printf '%s\n' "$cli_output" >&2
    report_typed_sandbox_failure "$cli_output" || true
    docker logs "$server_name" >&2 || true
    printf 'CLI did not reach the fake endpoint pause point\n' >&2
    return 1
  fi
}

case "$action" in
  run)
    set +e
    output=$(docker run --rm --name "$cli_name" "${common[@]}" --network "$network" \
      "${runtime_env[@]}" "$image" "${run_args[@]}" 2>&1)
    result=$?
    set -e
    printf '%s\n' "$output"
    if report_typed_sandbox_failure "$output"; then exit 1; fi
    if (( result != 0 )); then
      printf 'StateFrame run exited %s\n' "$result" >&2
      exit "$result"
    fi
    ;;
  sigterm)
    docker run --detach --name "$cli_name" --label "sf-eks-smoke.owner=$owner" "${common[@]}" --network "$network" \
      "${runtime_env[@]}" "$image" "${run_args[@]}" >/dev/null
    made_cli=1
    wait_for_pause
    docker stop --time 10 "$cli_name" >/dev/null
    exit_code=$(docker container inspect --format '{{.State.ExitCode}}' "$cli_name")
    if [[ "$exit_code" == 137 ]]; then
      printf 'SIGTERM did not stop the CLI before Docker sent SIGKILL\n' >&2
      exit 1
    fi
    printf 'SIGTERM stopped the CLI with exit code %s\n' "$exit_code"
    ;;
  kill-resume)
    docker run --detach --name "$cli_name" --label "sf-eks-smoke.owner=$owner" "${common[@]}" --network "$network" \
      "${runtime_env[@]}" "$image" "${run_args[@]}" >/dev/null
    made_cli=1
    wait_for_pause
    docker kill --signal=KILL "$cli_name" >/dev/null
    exit_code=$(docker container inspect --format '{{.State.ExitCode}}' "$cli_name")
    if [[ "$exit_code" != 137 ]]; then
      printf 'expected killed CLI exit 137, got %s\n' "$exit_code" >&2
      exit 1
    fi
    docker exec "$server_name" python -c \
      'from pathlib import Path; Path("/workspace/.server-state/abandon").write_text("killed request", encoding="utf-8")'
    set +e
    output=$(docker run --rm --name "$resume_name" "${common[@]}" --network "$network" \
      "${runtime_env[@]}" "$image" resume /home/sf/smoke-run 2>&1)
    result=$?
    set -e
    printf '%s\n' "$output"
    if report_typed_sandbox_failure "$output"; then exit 1; fi
    if (( result != 0 )); then
      printf 'StateFrame resume exited %s\n' "$result" >&2
      exit "$result"
    fi
    ;;
esac

if [[ "$action" != sigterm ]]; then
  docker run --rm --name "$test_name" "${common[@]}" --network none \
    --workdir /workspace/repo --entrypoint python "$image" -m pytest -q tests/test_calc.py
  docker run --rm --name "$diff_name" "${common[@]}" --network none \
    --mount "type=bind,source=$script_dir/tiny_repo,target=/baseline,readonly" \
    --entrypoint python "$image" -c \
    'from difflib import unified_diff; from pathlib import Path; old=Path("/baseline/calc.py").read_text().splitlines(True); new=Path("/workspace/repo/calc.py").read_text().splitlines(True); assert old != new and "    return 41\n" in old and "    return 42\n" in new; print("".join(unified_diff(old, new, fromfile="baseline/calc.py", tofile="workspace/calc.py")), end="")'
  printf 'smoke checks passed: StateFrame exit 0, independent pytest pass, edit diff present\n'
fi
