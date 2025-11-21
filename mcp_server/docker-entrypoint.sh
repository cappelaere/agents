#!/usr/bin/env bash
set -euo pipefail

# Entry point for the MCP server container.
# All configuration should be provided via environment variables
# (e.g., docker-compose env_file or `environment`), not via env.sh.

: "${START_CMD:?Set START_CMD to your ais mcp server start command}"
echo "[entrypoint] Starting with command: ${START_CMD}"
exec bash -lc "${START_CMD}"
