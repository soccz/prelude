#!/usr/bin/env bash
# Distribution beta cron entry — KST 09:05.
# Stage 1 = dry-run (telegram OFF, log only).
# 사용: '5 9 * * * cd /home/soccz/22tb/prelude && bash scripts/daily_run_distribution.sh'

set -euo pipefail

cd "$(dirname "$0")/.."
PROJ_ROOT="$(pwd)"

# venv
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# .env (NOT used for telegram in Stage 1, but harmless to load)
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

LOG_DIR="output"
mkdir -p "$LOG_DIR"
TODAY=$(date +%Y%m%d)
LOG="$LOG_DIR/cron_dist_$TODAY.log"

echo "=== prelude distribution beta KST $(date +%Y-%m-%d\ %H:%M:%S) ===" >> "$LOG"

# 1) Data update (incremental). 실패해도 health gate 가 잡음
echo "[1/3] data update — d1 + 4h all (close-out 가 4h 필요)" >> "$LOG"
python -m data.collector_d1 --update >> "$LOG" 2>&1 || echo "  d1 update warn" >> "$LOG"
# 4h ALL coins (close-out 이 모든 alert coin 의 어제 4h 봉 필요).
# --days 2 로 incremental (1-2 페이지/코인, 252 × ~0.5s ≈ 2분).
python -m data.collector_4h --all --days 2 >> "$LOG" 2>&1 || echo "  4h update warn" >> "$LOG"

# 2) Health gate — stale DB 로 paper entry 만들지 않음
echo "[2/3] health_check gate" >> "$LOG"
python scripts/health_check.py --no-telegram >> "$LOG" 2>&1

# 3) Distribution beta — Stage 2 (telegram ON, 사용자 명시 활성화)
echo "[3/3] predict_today_distribution (Stage 2 — telegram ON)" >> "$LOG"
python scripts/predict_today_distribution.py \
    --universe top100 \
    --top-k 10 \
    --send-telegram \
    >> "$LOG" 2>&1
EXIT=$?

# 1h/15m incremental update 는 별도 research cron 으로 분리 (Phase B/C 용, 운영 critical X)

echo "[done] $(date +%H:%M:%S) exit=$EXIT" >> "$LOG"
echo "" >> "$LOG"
exit $EXIT
