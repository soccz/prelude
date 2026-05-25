#!/usr/bin/env bash
# Compatibility wrapper.
# Current 09:05 operating path is distribution + ACTIVE-only Telegram.

set -euo pipefail

cd "$(dirname "$0")/.."
exec bash scripts/daily_run_distribution.sh "$@"
