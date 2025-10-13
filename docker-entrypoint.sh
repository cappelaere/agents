#!/usr/bin/env bash
set -euo pipefail

# Source environment variables
if [[ -f "./env.sh" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ./env.sh
  set +a
fi

: "${START_CMD:?Set START_CMD to your agent start command}"
echo "[entrypoint] Starting with command: ${START_CMD}"
exec bash -lc "${START_CMD}"
