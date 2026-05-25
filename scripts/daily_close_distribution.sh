#!/usr/bin/env bash
# Distribution paper/shadow ledger close-out entry.
#
# Runs after KST 09:00 daily candle boundary so yesterday's target day has
# final 4h bars. This keeps distribution paper ledger and shadow ledger aligned
# before the dashboard/idea-validation publish step.

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
LOG="$LOG_DIR/cron_close_$TODAY.log"

echo "=== prelude distribution close KST $(date +%Y-%m-%d\ %H:%M:%S) ===" >> "$LOG"

echo "[1/3] data update — d1 + 4h all 2 days" >> "$LOG"
python -m data.collector_d1 --update >> "$LOG" 2>&1 || echo "  d1 update warn" >> "$LOG"
python -m data.collector_4h --all --days 2 >> "$LOG" 2>&1 || echo "  4h update warn" >> "$LOG"

echo "[2/3] close_paper_ledger" >> "$LOG"
python scripts/close_paper_ledger.py >> "$LOG" 2>&1
EXIT=$?

echo "[3/4] train_recommendation_meta (shadow-gated)" >> "$LOG"
python scripts/train_recommendation_meta.py >> "$LOG" 2>&1 || echo "  recommendation meta train warn" >> "$LOG"

echo "[4/4] idea_validation_report" >> "$LOG"
python scripts/idea_validation_report.py >> "$LOG" 2>&1 || echo "  idea validation warn" >> "$LOG"
python scripts/build_idea_validation_html.py >> "$LOG" 2>&1 || echo "  idea validation html warn" >> "$LOG"

echo "[done] $(date +%H:%M:%S) exit=$EXIT" >> "$LOG"
echo "" >> "$LOG"
exit $EXIT
