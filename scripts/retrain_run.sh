#!/usr/bin/env bash
# 일요일 KST 06:00 cron — 주간 재학습 + promotion gate.
# 사용: cron '0 6 * * 0 cd /home/soccz/22tb/prelude && bash scripts/retrain_run.sh'

set -euo pipefail

cd "$(dirname "$0")/.."
if [ -d "venv" ]; then source venv/bin/activate; fi
if [ -f ".env" ]; then set -a; source .env; set +a; fi

LOG="output/cron_retrain_$(date +%Y%m%d).log"
mkdir -p output

echo "=== prelude retrain $(date) ===" >> "$LOG"

# 1. 데이터 incremental update (전체 코인 최근 7일)
echo "[1/3] data update" >> "$LOG"
python -m data.collector_d1 --update >> "$LOG" 2>&1 || true
python -m data.collector_4h --update >> "$LOG" 2>&1 || true
# binance d1 는 매주 한번 전체 update
python -m data.collector_binance_d1 --all --days 1095 >> "$LOG" 2>&1 || true

# 2. 재학습 + promotion gate
echo "[2/3] retrain pipeline" >> "$LOG"
python -m signals.retrain --n-trials 30 >> "$LOG" 2>&1
EXIT=$?

# 3. 끝 + 텔레그램 알림
echo "[3/3] notify" >> "$LOG"
python -c "
import json
from pathlib import Path
from notifier.telegram import send_telegram

hist = Path('output/retrain_history.json')
if hist.exists():
    with open(hist) as f:
        h = json.load(f)
    last = h[-1]
    promoted = last['promoted']
    icon = '✅' if promoted else '❌'
    new = last['new_metrics']
    old = last.get('old_metrics') or {}
    msg = (
        f'{icon} prelude 주간 재학습 ({last[\"tag\"]})\n'
        f'결과: {\"PROMOTED\" if promoted else \"REJECTED\"}\n'
        f'new acc {new[\"accuracy\"]:.3f} brier {new[\"brier\"]:.3f}\n'
    )
    if old:
        msg += f'old acc {old.get(\"accuracy\",0):.3f} brier {old.get(\"brier\",0):.3f}\n'
    if not promoted:
        msg += f'reasons: {last[\"reasons\"]}\n'
    send_telegram(msg)
" >> "$LOG" 2>&1 || true

echo "done $(date)" >> "$LOG"
echo "" >> "$LOG"
exit $EXIT
