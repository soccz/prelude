#!/usr/bin/env bash
# Pre-open trigger cron entry — KST 08:55.
# Telegram sends every day: ACTIVE recommendation or concise silence/status.

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
# 255 coin × ~1s ≈ 4분이라 5분 timeout 시 predict 못 도달 — --days 1 + service
# TimeoutStartSec=600 로 강화.
echo "[1/3] data update — d1 + 15m all 1 day" >> "$LOG"
python -m data.collector_d1 --update >> "$LOG" 2>&1 || echo "  d1 update warn" >> "$LOG"
python -m data.collector_15m_upbit --all --days 1 >> "$LOG" 2>&1 || echo "  15m update warn" >> "$LOG"

# 2) Health check
echo "[2/3] health_check gate (preopen: d1 + 15m)" >> "$LOG"
python scripts/health_check.py --channel preopen --no-telegram >> "$LOG" 2>&1 || echo "  health gate warn" >> "$LOG"

# 3) Pre-open trigger predict — RECORD ONLY (telegram OFF; R1 레이더가 발송 담당).
#    --no-telegram 으로 발송만 끄고 shadow/paper 기록은 유지(window-gate, 대시보드 continuity).
echo "[3/4] predict_preopen_trigger (record only — --no-telegram)" >> "$LOG"
python scripts/predict_preopen_trigger.py \
    --top-k 8 \
    --universe top100 \
    --no-telegram \
    >> "$LOG" 2>&1 || echo "  preopen predict warn (record only)" >> "$LOG"

# 4) R1 risk-reward 레이더 — 이 채널의 유일한 텔레그램 발송 (downside-first, SHADOW)
echo "[4/4] recommend_send (R1 radar, preopen slot)" >> "$LOG"
python scripts/recommend_send.py --slot preopen >> "$LOG" 2>&1
EXIT=$?

echo "[done] $(date +%H:%M:%S) exit=$EXIT" >> "$LOG"
echo "" >> "$LOG"
exit $EXIT
