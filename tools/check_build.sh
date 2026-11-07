#!/bin/bash
#
# tools/check_build.sh - check that your bot will pass the tournament build.
#
# Run it from the root of your repo (where agent.py is):
#     ./tools/check_build.sh
#
# It does what the organizers' build does, on your own computer:
#   1. builds the base image (docker/Dockerfile.base), the same recipe the
#      tournament uses. The first time this takes a while: it downloads
#      PyTorch. After that Docker reuses it.
#   2. builds YOUR bot on top of it (docker/Dockerfile.agent)
#   3. checks the size you added on top of the base (the cap is 3 GB)
#   4. starts your bot with the same restrictions as a real match (no
#      network, read-only files except /tmp, limited CPU and memory) and runs
#      the smoke test: it must start within 15 s and answer sample hands with
#      legal actions, in under 1 s each.
#
# Needs Docker (Docker Desktop on Mac/Windows; on Windows run this in WSL or
# Git Bash). Only files committed AND pushed to GitHub count in the real
# build, so commit your changes before you submit the commit SHA.
#
# Options (environment variables):
#     PLATFORM=linux/amd64 ./tools/check_build.sh
#         Build for the tournament's CPU type. Useful on Apple Silicon Macs to
#         catch packages that only exist for one CPU type. It is slower
#         there, so the timing checks may fail even if your bot is fine.
#     SMOKE_CPUS=1 SMOKE_MEMORY=4g    resources for the test (these defaults
#                                     match the tournament)

set -uo pipefail

# Git Bash on Windows rewrites arguments that look like Unix paths (/tmp,
# /smoke/smoke_test.py) into Windows paths before Docker sees them. Stop it.
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*'

# On Git Bash, "pwd -W" gives the Windows form (C:/...) that docker.exe understands.
ROOT="$(pwd -W 2>/dev/null || pwd)"
BASE_TAG="poker-agent-base:local"
BOT_TAG="poker-agent-check:latest"
MAX_ADDED_BYTES=3221225472            # 3 GiB, the tournament cap
SMOKE_CPUS="${SMOKE_CPUS:-1}"
SMOKE_MEMORY="${SMOKE_MEMORY:-4g}"
PLATFORM_ARGS=()
if [ -n "${PLATFORM:-}" ]; then
  PLATFORM_ARGS=(--platform "$PLATFORM")
fi
CONTAINER="poker-check-$$"

# Remove the test container when the script ends, however it ends.
cleanup() {
  docker rm -f "$CONTAINER" >/dev/null 2>&1
}
trap cleanup EXIT

fail() {
  echo
  echo "CHECK FAILED: $*"
  exit 1
}

step() {
  echo
  echo "==> $*"
}

# --- Are we in the right place? ------------------------------------------------
for f in agent.py requirements.txt docker/Dockerfile.base docker/Dockerfile.agent tools/smoke_test.py; do
  [ -f "$ROOT/$f" ] || fail "can't find $f. Run this from the root of your repo."
done
docker info >/dev/null 2>&1 || fail "Docker isn't running (start Docker Desktop, or the docker service)."

# --- 1. Base image -------------------------------------------------------------
step "Building the base image ($BASE_TAG). The first time takes a few minutes..."
# The base needs no files, so it is built from an empty context ("-" reads
# the Dockerfile from standard input).
docker build ${PLATFORM_ARGS[@]+"${PLATFORM_ARGS[@]}"} -t "$BASE_TAG" - < "$ROOT/docker/Dockerfile.base" \
  || fail "the base image did not build (this is not your bot's fault; tell the organizers)"

# --- 2. Your bot ---------------------------------------------------------------
step "Building your bot on top of the base image..."
docker build ${PLATFORM_ARGS[@]+"${PLATFORM_ARGS[@]}"} -f "$ROOT/docker/Dockerfile.agent" -t "$BOT_TAG" "$ROOT" \
  || fail "your bot's image did not build. Read the error above (often a typo or a bad version in requirements.txt)."

# --- 3. Size -------------------------------------------------------------------
step "Checking the size..."
bot_bytes=$(docker image inspect --format '{{.Size}}' "$BOT_TAG")
base_bytes=$(docker image inspect --format '{{.Size}}' "$BASE_TAG")
added=$(( bot_bytes - base_bytes ))
[ "$added" -lt 0 ] && added=0
echo "    your bot adds $(( added / 1048576 )) MB on top of the base (cap: $(( MAX_ADDED_BYTES / 1048576 )) MB)"
if [ "$added" -gt "$MAX_ADDED_BYTES" ]; then
  fail "too big. Remove unused libraries, use CPU-only versions (no [cuda] extras), and list big files you don't need (datasets, checkpoints) in .dockerignore."
fi

# --- 4. Smoke test -------------------------------------------------------------
step "Starting your bot like the tournament does, then running the smoke test..."
docker run --detach --name "$CONTAINER" \
  --network none \
  --read-only \
  --tmpfs /tmp:rw,nosuid,nodev,size=512m \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --pids-limit 256 \
  --cpus "$SMOKE_CPUS" \
  --memory "$SMOKE_MEMORY" \
  --env PORT=5008 \
  --env AGENT_TOKEN=local-check \
  --volume "$ROOT/tools/smoke_test.py:/smoke/smoke_test.py:ro" \
  "$BOT_TAG" >/dev/null || fail "could not start your bot's image"

if ! docker exec "$CONTAINER" python /smoke/smoke_test.py --port 5008 --startup 15; then
  echo
  echo "---- the last lines your bot printed ----"
  docker logs --tail 40 "$CONTAINER" 2>&1
  fail "the smoke test failed (see the messages above)"
fi

echo
echo "ALL CHECKS PASSED. Commit and push, then submit your full commit SHA (git rev-parse HEAD)."
