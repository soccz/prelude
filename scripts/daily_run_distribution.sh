#!/usr/bin/env bash
# Distribution beta cron entry — KST 09:05.
# Telegram sends every day: ACTIVE recommendation or concise silence/status.
# 사용: '5 9 * * * cd /home/soccz/22tb/prelude && bash scripts/daily_run_distribution.sh'

set -euo pipefail
export TZ=Asia/Seoul

cd "$(dirname "$0")/.."
PROJ_ROOT="$(pwd)"

# venv
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
LOG="$LOG_DIR/cron_dist_$TODAY.log"
if [ -L "$LOG" ] || { [ -e "$LOG" ] && [ ! -f "$LOG" ]; }; then
    echo "distribution log must be a regular non-symlink file: $LOG" >&2
    exit 2
fi

echo "=== prelude distribution beta KST $(date +%Y-%m-%d\ %H:%M:%S) ===" >> "$LOG"

EXIT=0
record_critical_failure() {
    local rc="$1"
    local step="$2"
    if [ "$EXIT" -eq 0 ]; then
        EXIT="$rc"
    fi
    echo "  [critical] $step failed (exit=$rc) — 가능한 후속 단계는 계속" >> "$LOG"
}

# R1은 D1만 필요하다. 느린 4h 수집/구형 distribution 예측보다 먼저 발송해
# 09:00 open 이후 불필요한 지연을 줄인다.
RECOMMEND_HEALTH_OK=0
DISTRIBUTION_HEALTH_OK=0
D1_UPDATE_OK=0
H4_UPDATE_OK=0

echo "[1/11] data update — d1" >> "$LOG"
if python -m data.collector_d1 --update >> "$LOG" 2>&1; then
    D1_UPDATE_OK=1
else
    record_critical_failure "$?" "D1 universe update"
    echo "  R1/R2/A1/PUMP send·record skipped (partial/failed D1 update)" >> "$LOG"
fi

# D1-only hard gate. 실패하면 stale R1/R2/A1/v2 생성은 건너뛰되 4h/Binance
# 유지보수와 후속 감사는 계속하고 최종 nonzero를 전파한다.
echo "[2/11] health_check gate (recommend: d1 only)" >> "$LOG"
if [ "$D1_UPDATE_OK" -eq 1 ]; then
    if python scripts/health_check.py --channel recommend --no-telegram >> "$LOG" 2>&1; then
        RECOMMEND_HEALTH_OK=1
    else
        record_critical_failure "$?" "recommend D1 health gate"
        echo "  R1/R2/A1/PUMP send·record skipped (stale D1)" >> "$LOG"
    fi
fi

# R1 risk-reward 레이더 — 텔레그램 발송 재개 (2026-07-18 사용자 지시, DECISIONS #2 부분 개정).
#    구 최소관심 강등(07-11 --dry-run)은 해제. ledger 기록(recommend_today)은 원래부터 dry-run과
#    무관하게 동일 — 09-01 사전등록 판정에 영향 0. 판정일·모라토리엄·v2 기준은 그대로 유지.
echo "[3/11] recommend_send + recommend_today (R1 open — 발송 재개 07-18)" >> "$LOG"
if [ "$RECOMMEND_HEALTH_OK" -eq 1 ]; then
    if python scripts/recommend_send.py --slot open >> "$LOG" 2>&1; then
        :
    else
        record_critical_failure "$?" "R1 open send"
    fi
    if python scripts/recommend_today.py --require-receipt >> "$LOG" 2>&1; then
        :
    else
        record_critical_failure "$?" "R1 ledger"
    fi
fi

# 4h ALL coins (close-out 이 모든 alert coin 의 어제 4h 봉 필요).
# --days 2 로 incremental (1-2 페이지/코인, 252 × ~0.5s ≈ 2분).
echo "[4/11] data update — 4h all" >> "$LOG"
if python -m data.collector_4h --all --days 2 >> "$LOG" 2>&1; then
    H4_UPDATE_OK=1
else
    record_critical_failure "$?" "4h universe update"
fi

echo "[5/11] health_check gate (distribution: d1 + 4h)" >> "$LOG"
if [ "$D1_UPDATE_OK" -eq 1 ] && [ "$H4_UPDATE_OK" -eq 1 ]; then
    if python scripts/health_check.py --channel distribution --no-telegram >> "$LOG" 2>&1; then
        DISTRIBUTION_HEALTH_OK=1
    else
        record_critical_failure "$?" "distribution health gate"
        echo "  distribution record skipped (stale d1/4h)" >> "$LOG"
    fi
else
    echo "  distribution record skipped (D1/4h update failed)" >> "$LOG"
fi

# Distribution beta — RECORD ONLY (telegram OFF; paper_ledger/대시보드 유지).
# send 플래그 제거 → dry_run, paper_ledger append 는 window-gate 라 그대로 기록됨.
echo "[6/11] predict_today_distribution (record only — no send flags)" >> "$LOG"
if [ "$DISTRIBUTION_HEALTH_OK" -eq 1 ]; then
    if python scripts/predict_today_distribution.py \
        --universe top100 \
        --top-k 10 \
        >> "$LOG" 2>&1; then
        :
    else
        record_critical_failure "$?" "distribution record-only prediction"
    fi
fi

# R2 challenger (downside-penalized) — SHADOW · record-only (텔레그램 X). forward 표본
#    적립용으로 자기 ledger(shadow_ledger_recommend_r2.csv)에만 기록. challenger_only=True 라
#    champion_selector 가 영구 차단(절대 발송 안 됨). 30거래일 후 R1 과 하방-우선 비교.
#    실패해도 R1 운영과 무관(가드).
echo "[7/11] recommend_today --ranking R2 (record-only)" >> "$LOG"
if [ "$RECOMMEND_HEALTH_OK" -eq 1 ]; then
    if python scripts/recommend_today.py --ranking R2 >> "$LOG" 2>&1; then
        :
    else
        record_critical_failure "$?" "R2 record-only ledger"
    fi
fi

# A1 sustainability challenger — SHADOW · record-only (텔레그램 X). R1 top-3 위에
#    dump head 로 dump-prone 픽 강등→교체. 자기 ledger(shadow_ledger_recommend_sustain.csv)
#    에만 기록. challenger_only=True 라 champion_selector 가 영구 차단(절대 발송 안 됨).
#    30거래일 후 R1 과 하방-우선 비교. 실패해도 R1 운영과 무관(가드).
echo "[8/11] recommend_today --ranking A1 (record-only)" >> "$LOG"
if [ "$RECOMMEND_HEALTH_OK" -eq 1 ]; then
    if python scripts/recommend_today.py --ranking A1 >> "$LOG" 2>&1; then
        :
    else
        record_critical_failure "$?" "A1 record-only ledger"
    fi
fi

# PUMP hunter rule detector — SHADOW · record-only (텔레그램 X). pump_rule_discovery_v1
#    에서 채굴한 D-1 roc_7d/ATR/log_return 룰을 매일 별도 ledger 에 기록한다.
#    challenger_only=True 라 champion_selector 가 발송 승격하지 않음. policy_competition 이
#    CLOSED forward rows 로 기존 모델들과 pump20 recall/net/downside 를 비교.
echo "[9/11] pump_detector_today (PUMP hunter ledger, record-only)" >> "$LOG"
if [ "$RECOMMEND_HEALTH_OK" -eq 1 ]; then
    if python scripts/pump_detector_today.py >> "$LOG" 2>&1; then
        :
    else
        record_critical_failure "$?" "PUMP hunter record-only ledger"
    fi
fi

# Binance d1 incremental refresh — v2 detector 의 b_vol_surge 용. Binance D-1 일봉은
#    00:00 UTC = KST 09:00 마감이라 이 시점 (09:10+) 에 fresh 하게 받는다.
#    메인 R1 알림 뒤라 실패/지연해도 기존 운영 무영향 (가드).
echo "[10/11] collector_binance_d1 --days 3 (v2 feature incremental)" >> "$LOG"
if python -m data.collector_binance_d1 --all --days 3 >> "$LOG" 2>&1; then
    :
else
    record_critical_failure "$?" "Binance D1 refresh"
fi

# PUMP hunter v2 — Binance volsurge 융합 radar. 사용자 컨펌 (2026-06-11) 으로
#    🎯 텔레그램 발사 (후보 있을 때만 / binance stale 시 경고 1줄 / 후보 0 + 정상 = 무소음).
#    shadow ledger 기록. champion 승격은 challenger_only 차단 — 별도 radar 채널.
echo "[11/11] pump_detector_v2_today (🎯 radar telegram + shadow ledger)" >> "$LOG"
if [ "$RECOMMEND_HEALTH_OK" -eq 1 ]; then
    if python scripts/pump_detector_v2_today.py --send-telegram >> "$LOG" 2>&1; then
        :
    else
        record_critical_failure "$?" "PUMP v2 attempted send/ledger"
    fi
else
    # fail-loud: 침묵 스킵 금지 — health gate 연쇄로 v2 가 못 나간 날을 기록
    echo "  [skip] PUMP v2 — recommend health gate fail 로 후보 생산 불가" >> "$LOG"
fi

# 1h/15m incremental update 는 별도 research cron 으로 분리 (Phase B/C 용, 운영 critical X)

echo "[done] $(date +%H:%M:%S) exit=$EXIT" >> "$LOG"
echo "" >> "$LOG"
exit "$EXIT"
