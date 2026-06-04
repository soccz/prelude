#!/usr/bin/env bash
# Pre-open paper ledger close-out entry.
#
# The 08:50 pre-open run updates 15m data before the 09:00 candle starts.
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

# ★ set -e 가드: close_preopen_ledger 실패해도 아래 meta-train/idea-validation 은 계속.
echo "[2/3] close_preopen_ledger" >> "$LOG"
if python scripts/close_preopen_ledger.py >> "$LOG" 2>&1; then
    EXIT=0
else
    EXIT=$?
    echo "  preopen close warn (exit=$EXIT) — meta/idea-validation 은 계속" >> "$LOG"
fi

echo "[3/4] train_recommendation_meta (shadow-gated)" >> "$LOG"
python scripts/train_recommendation_meta.py >> "$LOG" 2>&1 || echo "  recommendation meta train warn" >> "$LOG"

echo "[3b/4] policy_competition (model + send-policy forward audit)" >> "$LOG"
python -m ops.policy_competition >> "$LOG" 2>&1 || echo "  policy_competition warn" >> "$LOG"

echo "[4/4] idea_validation_report" >> "$LOG"
python scripts/idea_validation_report.py >> "$LOG" 2>&1 || echo "  idea validation warn" >> "$LOG"
python scripts/build_idea_validation_html.py >> "$LOG" 2>&1 || echo "  idea validation html warn" >> "$LOG"

echo "[done] $(date +%H:%M:%S) exit=$EXIT" >> "$LOG"
echo "" >> "$LOG"
exit $EXIT
