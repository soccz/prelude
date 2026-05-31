#!/usr/bin/env bash
# 추천 SHADOW ledger 청산 (09:35 KST = 00:35 UTC) — prior-day open 행 실현 채우기.
# prelude-recommend-close.timer 가 발사. 텔레그램 발송 X / 자동주문 X / 업비트 API key X.
# close_recommend_ledger.py 가 status=open 행을 -3%SL/+5%TP 15m 경로로 청산 +
# net realized(왕복 0.15% 차감) + pump20_hit 채움 → forward 평가 표본 누적.
# 사용: '35 9 * * * cd /home/soccz/22tb/prelude && bash scripts/daily_recommend_close.sh'
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -d "venv" ]; then
    source venv/bin/activate
fi
# .env (close 에는 텔레그램 토큰 불요지만 일관성 위해 로드 — 발송 코드 없음)
if [ -f ".env" ]; then
    set -a; source .env; set +a
fi

LOG_DIR="output"
mkdir -p "$LOG_DIR"
TODAY=$(date +%Y%m%d)
LOG="$LOG_DIR/cron_recommend_close_$TODAY.log"

echo "=== prelude recommend close (09:35 KST) $(date +%Y-%m-%d\ %H:%M:%S) ===" >> "$LOG"

# 1) 데이터 update — 청산 경로용 15m + 일봉 pump20용 d1 (전날 경로 완전 마감 보장).
echo "[1/2] data update — 15m all 2 days + d1" >> "$LOG"
python -m data.collector_15m_upbit --all --days 2 >> "$LOG" 2>&1 || echo "  15m update warn" >> "$LOG"
python -m data.collector_d1 --update >> "$LOG" 2>&1 || echo "  d1 update warn" >> "$LOG"

# 2) recommend SHADOW ledger 청산 (기록만 — 발송/주문 없음). cutoff = asof-1day.
echo "[2/2] close_recommend_ledger" >> "$LOG"
python scripts/close_recommend_ledger.py >> "$LOG" 2>&1
EXIT=$?

echo "[done] $(date +%H:%M:%S) exit=$EXIT" >> "$LOG"
echo "" >> "$LOG"
exit $EXIT
