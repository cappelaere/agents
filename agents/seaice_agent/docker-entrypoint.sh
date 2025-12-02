#!/usr/bin/env bash
set -euo pipefail

: "${START_CMD:?Set START_CMD to your seaice agent start command}"
echo "[entrypoint] Starting with command: ${START_CMD}"
exec bash -lc "${START_CMD}"
