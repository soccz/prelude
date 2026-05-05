#!/usr/bin/env bash
# Pre-open trigger cron entry — KST 08:55.
# Stage 2 (telegram ON, 사용자 명시 활성화).

set -euo pipefail

cd "$(dirname "$0")/.."
PROJ_ROOT="$(pwd)"

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
LOG="$LOG_DIR/cron_preopen_$TODAY.log"

echo "=== prelude pre-open trigger KST $(date +%Y-%m-%d\ %H:%M:%S) ===" >> "$LOG"

# 1) Data update (d1 + 15m incremental). 15m 가 핵심 — 08:30 snapshot 포함되어야 함.
echo "[1/3] data update — d1 + 15m all 2 days" >> "$LOG"
python -m data.collector_d1 --update >> "$LOG" 2>&1 || echo "  d1 update warn" >> "$LOG"
python -m data.collector_15m_upbit --all --days 2 >> "$LOG" 2>&1 || echo "  15m update warn" >> "$LOG"

# 2) Health check
echo "[2/3] health_check gate" >> "$LOG"
python scripts/health_check.py --no-telegram >> "$LOG" 2>&1 || echo "  health gate warn" >> "$LOG"

# 3) Pre-open trigger predict + telegram
echo "[3/3] predict_preopen_trigger (Stage 2 — telegram ON)" >> "$LOG"
python scripts/predict_preopen_trigger.py \
    --top-k 8 \
    --universe top100 \
    >> "$LOG" 2>&1
EXIT=$?

echo "[done] $(date +%H:%M:%S) exit=$EXIT" >> "$LOG"
echo "" >> "$LOG"
exit $EXIT
