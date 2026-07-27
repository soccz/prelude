#!/usr/bin/env bash
# Pre-open trigger cron entry — KST 08:50 (timer: 23:50 UTC).
# Telegram: R1 risk-reward radar via recommend_send --slot preopen (champion-aware).

set -euo pipefail
export TZ=Asia/Seoul

cd "$(dirname "$0")/.."
PROJ_ROOT="$(pwd)"

if [ -d "venv" ]; then
    source venv/bin/activate
fi

if [ -e "$PROJ_ROOT/.env" ] || [ -L "$PROJ_ROOT/.env" ]; then
    # shellcheck disable=SC1091
    source "$PROJ_ROOT/deploy/load_runtime_env.sh"
    RUNTIME_PYTHON="$(command -v python)" || {
        echo "Python runtime unavailable" >&2
        exit 2
    }
    load_prelude_runtime_env "$PROJ_ROOT/.env" "$RUNTIME_PYTHON"
fi

LOG_DIR="output"
mkdir -p "$LOG_DIR"
if [ -L "$LOG_DIR" ] || [ ! -d "$LOG_DIR" ]; then
    echo "output must be a real directory: $LOG_DIR" >&2
    exit 2
fi
TODAY=$(date +%Y%m%d)
LOG="$LOG_DIR/cron_preopen_$TODAY.log"
if [ -L "$LOG" ] || { [ -e "$LOG" ] && [ ! -f "$LOG" ]; }; then
    echo "pre-open log must be a regular non-symlink file: $LOG" >&2
    exit 2
fi

echo "=== prelude pre-open trigger KST $(date +%Y-%m-%d\ %H:%M:%S) ===" >> "$LOG"

EXIT=0
D1_UPDATE_OK=0
RECOMMEND_HEALTH_OK=0
M15_UPDATE_OK=0
PREOPEN_HEALTH_OK=0
record_critical_failure() {
    local rc="$1"
    local step="$2"
    if [ "$EXIT" -eq 0 ]; then
        EXIT="$rc"
    fi
    echo "  [critical] $step failed (exit=$rc) — 가능한 후속 단계는 계속" >> "$LOG"
}

# R1은 D1만 사용한다. record-only legacy 15m 수집보다 먼저 검증·발송해
# 08:50 알림이 느린 전체-market 수집이나 희소 15m 봉에 막히지 않게 한다.
echo "[1/6] data update — d1" >> "$LOG"
if python -m data.collector_d1 --update >> "$LOG" 2>&1; then
    D1_UPDATE_OK=1
else
    rc=$?
    record_critical_failure "$rc" "preopen D1 universe update"
    echo "  R1 preopen send·record skipped (partial/failed D1 update)" >> "$LOG"
fi

# R1 preopen 전용 D1 gate: D-2 quote volume로 정한 PIT Top100 모두에
# 마지막 종가(D-1 09:00 시작 일봉)가 정확히 존재해야 한다.
echo "[2/6] health_check gate (recommend-preopen: d1 only)" >> "$LOG"
if [ "$D1_UPDATE_OK" -eq 1 ]; then
    if python scripts/health_check.py \
        --channel recommend-preopen \
        --no-telegram \
        >> "$LOG" 2>&1; then
        RECOMMEND_HEALTH_OK=1
    else
        record_critical_failure "$?" "R1 preopen D1 health gate"
        echo "  R1 preopen send·record skipped (stale D1)" >> "$LOG"
    fi
fi

# R1 risk-reward 레이더 — 텔레그램 발송 재개 (2026-07-18 사용자 지시,
# DECISIONS #2 부분 개정). 구 최소관심 강등(07-11 --dry-run)은 해제.
echo "[3/6] recommend_send + recommend_today (R1 preopen 전용 ledger)" >> "$LOG"
if [ "$RECOMMEND_HEALTH_OK" -eq 1 ]; then
    if python scripts/recommend_send.py --slot preopen >> "$LOG" 2>&1; then
        :
    else
        record_critical_failure "$?" "R1 preopen send"
    fi
    if python scripts/recommend_today.py --slot preopen --require-receipt \
        >> "$LOG" 2>&1; then
        :
    else
        record_critical_failure "$?" "R1 preopen ledger"
    fi
fi

# record-only legacy precursor용 15m incremental. 08:30 closed snapshot이
# 필요하며, 실패해도 이미 검증된 D1-only R1 발송을 되돌리지는 않는다.
echo "[4/6] data update — 15m all 1 day (record-only legacy)" >> "$LOG"
if python -m data.collector_15m_upbit --all --days 1 >> "$LOG" 2>&1; then
    M15_UPDATE_OK=1
else
    record_critical_failure "$?" "preopen 15m update"
fi

echo "[5/6] health_check gate (legacy preopen: d1 + 15m)" >> "$LOG"
if [ "$D1_UPDATE_OK" -eq 1 ] && [ "$M15_UPDATE_OK" -eq 1 ]; then
    if python scripts/health_check.py \
        --channel preopen \
        --no-telegram \
        >> "$LOG" 2>&1; then
        PREOPEN_HEALTH_OK=1
    else
        record_critical_failure "$?" "legacy preopen health gate"
        echo "  legacy preopen record skipped (stale D1/15m)" >> "$LOG"
    fi
else
    echo "  legacy preopen record skipped (D1/15m update failed)" >> "$LOG"
fi

# 구 pre-open trigger는 RECORD ONLY. --no-telegram으로 발송을 끄고
# shadow/paper continuity만 유지한다.
echo "[6/6] predict_preopen_trigger (record only — --no-telegram)" >> "$LOG"
if [ "$PREOPEN_HEALTH_OK" -eq 1 ]; then
    if python scripts/predict_preopen_trigger.py \
        --top-k 8 \
        --universe top100 \
        --no-telegram \
        >> "$LOG" 2>&1; then
        :
    else
        record_critical_failure "$?" "preopen record-only prediction"
    fi
fi

echo "[done] $(date +%H:%M:%S) exit=$EXIT" >> "$LOG"
echo "" >> "$LOG"
exit "$EXIT"
