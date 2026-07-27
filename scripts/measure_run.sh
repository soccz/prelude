#!/usr/bin/env bash
# 매일 KST 00:00 cron — IC + drift 측정 (어제 ledger 청산 후).
# 사용: cron '0 0 * * * cd /home/soccz/22tb/prelude && bash scripts/measure_run.sh'

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

LOG="output/cron_measure_$(date +%Y%m%d).log"
mkdir -p output

echo "=== prelude measure $(date) ===" >> "$LOG"

if python -c "
from pathlib import Path
import json
import pandas as pd
from ops.drift_detector import evaluate_drift, daily_ic
from ledger.metrics import compute_summary

# 1. ledger 로 hit_history 만들기
ledger_path = Path('output/ledger.csv')
if ledger_path.exists():
    df = pd.read_csv(ledger_path)
    df['entry_date'] = pd.to_datetime(df['entry_date']).dt.normalize()
    daily_hit = df.groupby('entry_date').apply(
        lambda g: (g['exit_type'] == 'tp').mean()
    )
    print(f'  ledger rows: {len(df)}, daily hit history: {len(daily_hit)}')

    # 2. IC history (signal_p_ge_10 vs actual exit_type tp)
    ic_path = Path('output/ic_history.json')
    ic_history = pd.Series(dtype=float)
    if 'signal_p_ge_10' in df.columns:
        df['actual_hit'] = (df['exit_type'] == 'tp').astype(int)
        ic_per_date = df.groupby('entry_date').apply(
            lambda g: g[['signal_p_ge_10', 'actual_hit']].corr(method='spearman').iloc[0,1]
            if len(g) >= 3 else float('nan')
        )
        ic_history = ic_per_date.dropna()

    # 3. drift 측정
    state = evaluate_drift(ic_history=ic_history, hit_history=daily_hit)
    print(f'  drift state: {state.state}, triggers: {state.triggers}')
    print(f'  details: {state.details}')

    # 4. 텔레그램 alert (WARN/FREEZE 시)
    if state.state != 'OK':
        from notifier.telegram import send_telegram
        from notifier.format import format_drift_alert
        msg = format_drift_alert(state)
        if not send_telegram(msg):
            raise RuntimeError('drift alert delivery failed')
else:
    print('  no ledger yet')
" >> "$LOG" 2>&1; then
    :
else
    RC=$?
    echo "  measure failed (exit=$RC)" >> "$LOG"
    exit "$RC"
fi

echo "done $(date)" >> "$LOG"
echo "" >> "$LOG"
