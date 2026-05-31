#!/usr/bin/env bash
# 추천 레이더 pre-open (08:50 KST = 23:50 UTC) — SHADOW(검증중) 텔레그램 발송.
# prelude-recommend-am.timer 가 발사. 자동주문 X / 업비트 API key X.
# 사용: '50 8 * * * cd /home/soccz/22tb/prelude && bash scripts/daily_recommend_am.sh'
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -d "venv" ]; then
    source venv/bin/activate
fi
# .env (TELEGRAM_BOT_TOKEN / CHAT_ID)
if [ -f ".env" ]; then
    set -a; source .env; set +a
fi

LOG_DIR="output"
mkdir -p "$LOG_DIR"
TODAY=$(date +%Y%m%d)
LOG="$LOG_DIR/cron_recommend_am_$TODAY.log"

echo "=== prelude recommend pre-open (08:50 KST) $(date +%Y-%m-%d\ %H:%M:%S) ===" >> "$LOG"

# 1) 데이터 incremental update (d1). stale 면 score_candidates 가 RuntimeError → 로그.
echo "[1/2] data update — d1" >> "$LOG"
python -m data.collector_d1 --update >> "$LOG" 2>&1 || echo "  d1 update warn" >> "$LOG"

# 2) risk-reward 레이더 발송 (SHADOW). 기록(ledger)은 recommend_today.py 의 책임.
echo "[2/2] recommend_send (preopen slot, telegram)" >> "$LOG"
python scripts/recommend_send.py --slot preopen >> "$LOG" 2>&1
EXIT=$?

echo "[done] $(date +%H:%M:%S) exit=$EXIT" >> "$LOG"
echo "" >> "$LOG"
exit $EXIT
