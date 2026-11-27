#!/usr/bin/env bash
# Run a command with the variables of the main checkout's environment file exported.
#
# The paid steps of the experiments read their API keys from the environment, and
# the background launcher forwards only the tracking variables into a run. This
# wrapper loads the shared file inside the run's own shell, so no key appears on
# a command line or in a log.
#
# Args:
#   <command...>     The command to run, with its own arguments.
#   SLM4IE_ENV_FILE  Optional environment variable: the file to load in place of
#                    the main checkout's .env.
#
# Example:
#   scripts/with_env.sh uv run --group evaluation python scripts/screen_silver_llms.py make --dry-run
set -euo pipefail

env_file="${SLM4IE_ENV_FILE:-$(git -C "$(dirname "$0")" worktree list --porcelain | awk 'NR==1 {print $2}')/.env}"
[ -f "$env_file" ] || { echo "with_env: no environment file at $env_file" >&2; exit 1; }
set -a
# shellcheck disable=SC1090
. "$env_file"
set +a
exec "$@"
