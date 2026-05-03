#!/usr/bin/env bash
# 매일 KST 09:30 cron — 어제 가상 ledger 결과 검증 + drift 측정.
# (실제 진입 시뮬은 predict_today.py 가 다음날 09시 시뮬로 처리하니 여기는 검증용)
# 사용: cron '30 9 * * * cd /home/soccz/22tb/prelude && bash scripts/post_open_run.sh'

set -euo pipefail

cd "$(dirname "$0")/.."
PROJ_ROOT="$(pwd)"

if [ -d "venv" ]; then source venv/bin/activate; fi
if [ -f ".env" ]; then set -a; source .env; set +a; fi

LOG="output/cron_post_open_$(date +%Y%m%d).log"
mkdir -p output

echo "=== prelude post_open KST $(date +%H:%M:%S) ===" >> "$LOG"

# 1. 텔레그램 ↔ ledger 일관성 검증
if [ -f "scripts/verify_telegram.py" ]; then
    echo "[1/2] verify telegram <-> ledger" >> "$LOG"
    python scripts/verify_telegram.py >> "$LOG" 2>&1 || echo "  verify warn" >> "$LOG"
fi

# 2. risk 상태 갱신
echo "[2/2] risk evaluate" >> "$LOG"
python -c "
from ledger.risk import evaluate_risk
from ledger.metrics import compute_summary
import pandas as pd
from pathlib import Path
ledger_path = Path('output/ledger.csv')
if ledger_path.exists():
    df = pd.read_csv(ledger_path)
    summary = compute_summary(df)
    state = evaluate_risk(
        daily_pnl_pct=summary.get('cum_return_pct', 0),  # 단순 누적 사용
        current_mdd_pct=summary.get('max_drawdown_pct', 0),
    )
    print(f'risk state: {state.state}, triggers: {state.triggers}')
" >> "$LOG" 2>&1

echo "done $(date +%H:%M:%S)" >> "$LOG"
echo "" >> "$LOG"
