#!/usr/bin/env bash
# Pre-open paper ledger close-out entry.
#
# The 08:55 pre-open run updates 15m data before the 09:00 candle starts.
# Close-out runs after 10:00 so first15/first30/first1h realized fields are
# all final, not partial.

set -euo pipefail

cd "$(dirname "$0")/.."

if [ -d "venv" ]; then
    source venv/bin/activate
fi

if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

LOG_DIR="output"
mkdir -p "$LOG_DIR"
TODAY=$(date +%Y%m%d)
LOG="$LOG_DIR/cron_preopen_close_$TODAY.log"

echo "=== prelude pre-open close KST $(date +%Y-%m-%d\ %H:%M:%S) ===" >> "$LOG"

echo "[1/2] data update — 15m all 1 day" >> "$LOG"
python -m data.collector_15m_upbit --all --days 1 >> "$LOG" 2>&1 || echo "  15m update warn" >> "$LOG"

echo "[2/2] close_preopen_ledger" >> "$LOG"
python scripts/close_preopen_ledger.py >> "$LOG" 2>&1
EXIT=$?

echo "[done] $(date +%H:%M:%S) exit=$EXIT" >> "$LOG"
echo "" >> "$LOG"
exit $EXIT
