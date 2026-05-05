#!/usr/bin/env bash
# 매일 KST 09:05 cron entry — 추론 + 텔레그램 알림.
# 사용: cron 등록 → '5 9 * * * cd /home/soccz/22tb/prelude && bash scripts/daily_run.sh'

set -euo pipefail

cd "$(dirname "$0")/.."
PROJ_ROOT="$(pwd)"

# venv 활성화
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# .env 자동 로드 (telegram.py 가 처리하지만 export 도)
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

LOG_DIR="output"
mkdir -p "$LOG_DIR"
TODAY=$(date +%Y%m%d)
LOG="$LOG_DIR/cron_daily_$TODAY.log"

echo "=== prelude daily_run KST $(date +%H:%M:%S) ==="  >> "$LOG"

# 1. 데이터 갱신 (어제 일봉 incremental)
echo "[1/3] data update" >> "$LOG"
python -m data.collector_d1 --update >> "$LOG" 2>&1 || echo "  d1 update warn" >> "$LOG"
python -m data.collector_4h --coin KRW-BTC --days 7 >> "$LOG" 2>&1 || true
# (binance d1 도 매일은 안 — 주간 retrain 시 갱신)

# 2. 추론 + 텔레그램
echo "[2/3] predict + telegram" >> "$LOG"
python scripts/predict_today.py >> "$LOG" 2>&1
EXIT=$?

if [ $EXIT -ne 0 ]; then
    echo "  predict_today exit $EXIT" >> "$LOG"
fi

# 3. 끝
echo "[3/3] done $(date +%H:%M:%S)" >> "$LOG"
echo "" >> "$LOG"
exit $EXIT
