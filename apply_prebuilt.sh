#!/usr/bin/env bash
set -Eeuo pipefail
export CI=true COMPOSE_MENU=false COMPOSE_PROFILES=""
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$ROOT"
COMPOSE=(docker compose --project-directory "$ROOT" -f "$ROOT/docker-compose.yaml")
IMAGE=""
PROXY_IP=""
WAIT_TIMEOUT="${DOCKER_WAIT_TIMEOUT_SEC:-300}"
UP_TIMEOUT="${DOCKER_UP_TIMEOUT_SEC:-1800}"
usage() {
    echo 'Usage: bash apply_prebuilt.sh --image=whisper-webui=sha256:ID --proxy-ip=PRIVATE_IPV4'
    echo 'Cold apply only: no build, pull, push, model operations, or automatic rollback.'
}
for arg in "$@"; do
    case "$arg" in
        --image=*) [[ -z "$IMAGE" ]] || { echo 'ERROR: duplicate image' >&2; exit 2; }; IMAGE="${arg#--image=}" ;;
        --proxy-ip=*) [[ -z "$PROXY_IP" ]] || { echo 'ERROR: duplicate proxy IP' >&2; exit 2; }; PROXY_IP="${arg#--proxy-ip=}" ;;
        --help|-h) usage; exit 0 ;;
        *) echo 'ERROR: unsupported argument' >&2; usage >&2; exit 2 ;;
    esac
done
[[ "$IMAGE" =~ ^whisper-webui=sha256:[0-9a-f]{64}$ && -n "$PROXY_IP" ]] || {
    echo 'ERROR: one expected Whisper image and verified private proxy IP are required' >&2; exit 2;
}
for value in "$WAIT_TIMEOUT" "$UP_TIMEOUT"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo 'ERROR: invalid timeout' >&2; exit 2; }
done
for command in git docker python3 timeout flock sha256sum; do
    command -v "$command" >/dev/null || { echo 'ERROR: required command missing' >&2; exit 1; }
done
python3 -c '
import ipaddress,sys
try: a=ipaddress.IPv4Address(sys.argv[1])
except ValueError: sys.exit(2)
networks=("10.0.0.0/8","172.16.0.0/12","192.168.0.0/16")
sys.exit(0 if any(a in ipaddress.IPv4Network(n) for n in networks) else 2)
' "$PROXY_IP" || {
    echo 'ERROR: proxy must be one exact private IPv4 address, not wildcard, CIDR or list' >&2; exit 2;
}
# Every release caller must lock this same inode; never unlink a held lock.
exec 9>>"$ROOT/.build.lock"
flock -n 9 || { echo 'ERROR: another Whisper deployment is running' >&2; exit 1; }
require_release_tree() {
    [[ "$(timeout --kill-after=10s 30s git symbolic-ref --short HEAD)" == master ]] || {
        echo 'ERROR: apply requires the canonical master branch' >&2; exit 1;
    }
    local status
    status="$(timeout --kill-after=10s 30s git status --porcelain --untracked-files=all)"
    [[ -z "$status" ]] || { echo 'ERROR: apply requires a clean worktree' >&2; exit 1; }
}
# Apply the sealed ID itself, not a mutable tag resolved after inspection.
export WHISPER_PREBUILT_IMAGE="${IMAGE#*=}"
require_release_tree
REVISION="$(timeout --kill-after=10s 30s git rev-parse HEAD)"
timeout --kill-after=10s 30s docker info >/dev/null
timeout --kill-after=10s 30s "${COMPOSE[@]}" config --quiet
# Configuration carries credentials: keep the render only in the pipe.
CONFIG_DIGEST="$(timeout --kill-after=10s 30s "${COMPOSE[@]}" config --format json | sha256sum)"
check_image() {
    local tag actual
    tag="$(timeout --kill-after=10s 30s "${COMPOSE[@]}" config --format json | python3 -c '
import json,sys
c=json.load(sys.stdin); s=c["services"]["whisper-webui"]
if s.get("environment",{}).get("FORWARDED_ALLOW_IPS") != "127.0.0.1,"+sys.argv[1]:
    raise SystemExit("ERROR: resolved proxy trust does not match the sealed private hop")
if s.get("image") != sys.argv[2]:
    raise SystemExit("ERROR: configured image is not the sealed immutable ID")
print(s["image"])
' "$PROXY_IP" "$WHISPER_PREBUILT_IMAGE")"
    actual="$(timeout --kill-after=10s 30s docker image inspect --format '{{.Id}}' "$tag")"
    [[ "$actual" == "${IMAGE#*=}" ]] || { echo 'ERROR: image identity mismatch; no apply' >&2; exit 1; }
}
check_image
require_release_tree
[[ "$(timeout --kill-after=10s 30s git rev-parse HEAD)" == "$REVISION" ]] || {
    echo 'ERROR: HEAD changed; no apply' >&2; exit 1;
}
[[ "$(timeout --kill-after=10s 30s "${COMPOSE[@]}" config --format json | sha256sum)" == "$CONFIG_DIGEST" ]] || {
    echo 'ERROR: resolved configuration changed; no apply' >&2; exit 1;
}
# Final gates precede the last ID check; a retag during those gates must fail.
check_image
timeout --kill-after=30s "${UP_TIMEOUT}s" "${COMPOSE[@]}" up -d --no-build --pull never \
    --wait --wait-timeout "$WAIT_TIMEOUT" --timeout 120 whisper-webui
echo 'Prebuilt Whisper apply returned; this is not HTTPS, job-drain or release acceptance.'
