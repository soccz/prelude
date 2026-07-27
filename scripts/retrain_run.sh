#!/usr/bin/env bash
# 일요일 KST 06:00 cron — 주간 재학습 + promotion gate.
# 사용: cron '0 6 * * 0 cd /home/soccz/22tb/prelude && bash scripts/retrain_run.sh'

set -euo pipefail

cd "$(dirname "$0")/.."
PROJ_ROOT="$(pwd)"
if [ -d "venv" ]; then source venv/bin/activate; fi
if [ -e "$PROJ_ROOT/.env" ] || [ -L "$PROJ_ROOT/.env" ]; then
    # shellcheck disable=SC1091
    source "$PROJ_ROOT/deploy/load_runtime_env.sh"
    RUNTIME_PYTHON="$(command -v python)" || {
        echo "Python runtime unavailable" >&2
        exit 2
    }
    load_prelude_runtime_env "$PROJ_ROOT/.env" "$RUNTIME_PYTHON"
fi

LOG="output/cron_retrain_$(date +%Y%m%d).log"
mkdir -p output

echo "=== prelude retrain $(date) ===" >> "$LOG"

# 1. 데이터 incremental update (전체 코인 최근 7일)
echo "[1/3] data update" >> "$LOG"
DATA_EXIT=0
record_data_failure() {
    local rc="$1"
    local source="$2"
    if [ "$DATA_EXIT" -eq 0 ]; then
        DATA_EXIT="$rc"
    fi
    echo "  data update failed: $source (exit=$rc)" >> "$LOG"
}
if python -m data.collector_d1 --update >> "$LOG" 2>&1; then
    :
else
    record_data_failure "$?" "upbit_d1"
fi
if python -m data.collector_4h --update >> "$LOG" 2>&1; then
    :
else
    record_data_failure "$?" "upbit_4h"
fi
# binance d1 는 매주 한번 전체 update
if python -m data.collector_binance_d1 --all --days 1095 >> "$LOG" 2>&1; then
    :
else
    record_data_failure "$?" "binance_d1"
fi
if [ "$DATA_EXIT" -ne 0 ]; then
    echo "[blocked] retrain skipped because one or more data updates failed" >> "$LOG"
    exit "$DATA_EXIT"
fi

# 2. 재학습 + promotion gate
echo "[2/3] retrain pipeline" >> "$LOG"
if python -m signals.retrain --n-trials 30 >> "$LOG" 2>&1; then
    :
else
    RETRAIN_EXIT=$?
    echo "  retrain pipeline failed (exit=$RETRAIN_EXIT)" >> "$LOG"
    exit "$RETRAIN_EXIT"
fi

# 3. 끝 + 텔레그램 알림
echo "[3/3] notify" >> "$LOG"
if python -c "
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
    if not send_telegram(msg):
        raise RuntimeError('retrain result delivery failed')
" >> "$LOG" 2>&1; then
    :
else
    NOTIFY_EXIT=$?
    echo "  retrain notification failed (exit=$NOTIFY_EXIT)" >> "$LOG"
    exit "$NOTIFY_EXIT"
fi

echo "done $(date)" >> "$LOG"
echo "" >> "$LOG"
