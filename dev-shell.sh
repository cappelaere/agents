#!/usr/bin/env bash

# Lightweight project-local shell setup for the agents repo.
# Intentionally avoids `set -euo pipefail` so that a minor error
# (e.g. missing venv or .env) does not kill the VS Code terminal.

cd /Users/patrice/Development/agents || return

# Activate venv only if not already active
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
fi

# Load .env (KEY=VALUE format)
if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  . .env
  set +a
fi
source ~/.bashrc
nvm use 24
export PS1="venv> "
echo "[dev-shell] venv and .env (if present) are loaded."