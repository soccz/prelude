#!/usr/bin/env bash
# Compatibility wrapper.
# Current 09:30 close path is distribution paper/shadow ledger close-out.

set -euo pipefail

cd "$(dirname "$0")/.."
exec bash scripts/daily_close_distribution.sh "$@"
