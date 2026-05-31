#!/usr/bin/env bash
# 추천 레이더 open (09:05 KST = 00:05 UTC) — SHADOW(검증중) 텔레그램 발송 + 기록.
# prelude-recommend-open.timer 가 발사. 자동주문 X / 업비트 API key X.
# 사용: '5 9 * * * cd /home/soccz/22tb/prelude && bash scripts/daily_recommend_open.sh'
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -d "venv" ]; then
    source venv/bin/activate
fi
if [ -f ".env" ]; then
    set -a; source .env; set +a
fi

LOG_DIR="output"
mkdir -p "$LOG_DIR"
TODAY=$(date +%Y%m%d)
LOG="$LOG_DIR/cron_recommend_open_$TODAY.log"

echo "=== prelude recommend open (09:05 KST) $(date +%Y-%m-%d\ %H:%M:%S) ===" >> "$LOG"

# 1) 데이터 incremental update (d1) — 09:00 봉 확정 반영.
echo "[1/3] data update — d1" >> "$LOG"
python -m data.collector_d1 --update >> "$LOG" 2>&1 || echo "  d1 update warn" >> "$LOG"

# 2) risk-reward 레이더 발송 (SHADOW, open slot).
echo "[2/3] recommend_send (open slot, telegram)" >> "$LOG"
python scripts/recommend_send.py --slot open >> "$LOG" 2>&1
SEND_EXIT=$?

# 3) shadow ledger 기록 (idempotent — 같은 날 중복 append 방지). 발송과 분리된 책임.
echo "[3/3] recommend_today (shadow ledger append)" >> "$LOG"
python scripts/recommend_today.py >> "$LOG" 2>&1 || echo "  ledger append warn" >> "$LOG"

echo "[done] $(date +%H:%M:%S) send_exit=$SEND_EXIT" >> "$LOG"
echo "" >> "$LOG"
exit $SEND_EXIT
