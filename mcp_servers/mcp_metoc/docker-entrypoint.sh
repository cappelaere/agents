#!/usr/bin/env bash
set -euo pipefail

: "${START_CMD:=python -m mcp_metoc.app}"
echo "[mcp_metoc] starting: ${START_CMD}"
exec bash -lc "${START_CMD}"

