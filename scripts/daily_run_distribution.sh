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

# Upbit의 새 09:00 D1 candle이 일부 market에서 첫 fetch 직후 잠시 보이지
# 않는 경우가 있다. health gate 자체는 그대로 fail-closed로 두고, 실패 시
# 현재 경계가 없는 live market만 각 health gate에서 한 번 재수집한 뒤
# 같은 gate를 한 번 더 평가한다. recommend 검사 뒤 새 거래가 시작되면
# distribution 검사에서 결손 집합이 달라질 수 있으므로 두 gate의 상한을
# 분리한다. 무한 retry나 stale 관용은 금지한다.
# 지연 2초는 07-30 관측
# (첫 수집 직후 부재, health 시점 업스트림 존재)의 초기값이며 로그로 조정한다.
D1_RECONCILE_DELAY_SECONDS="${PRELUDE_D1_RECONCILE_DELAY_SECONDS:-2}"
if ! [[ "$D1_RECONCILE_DELAY_SECONDS" =~ ^[0-9]+$ ]]; then
    echo "invalid PRELUDE_D1_RECONCILE_DELAY_SECONDS: $D1_RECONCILE_DELAY_SECONDS" >&2
    exit 2
fi
RECOMMEND_D1_RECONCILE_ATTEMPTED=0
DISTRIBUTION_D1_RECONCILE_ATTEMPTED=0

run_health_with_d1_reconcile() {
    local channel="$1"
    local first_rc
    local refresh_rc=0
    local final_rc
    local reconcile_attempted

    case "$channel" in
        recommend)
            reconcile_attempted="$RECOMMEND_D1_RECONCILE_ATTEMPTED"
            ;;
        distribution)
            reconcile_attempted="$DISTRIBUTION_D1_RECONCILE_ATTEMPTED"
            ;;
        *)
            echo "  invalid health channel for D1 reconcile: $channel" >> "$LOG"
            return 2
            ;;
    esac

    if python scripts/health_check.py \
        --channel "$channel" --no-telegram >> "$LOG" 2>&1; then
        return 0
    else
        first_rc=$?
    fi
    if [ "$reconcile_attempted" -eq 1 ]; then
        echo "  health gate failed after its D1 reconcile was already used" \
            "(channel=$channel exit=$first_rc) — 추가 retry 없이 fail-closed" \
            >> "$LOG"
        return "$first_rc"
    fi
    case "$channel" in
        recommend)
            RECOMMEND_D1_RECONCILE_ATTEMPTED=1
            ;;
        distribution)
            DISTRIBUTION_D1_RECONCILE_ATTEMPTED=1
            ;;
    esac
    echo "  health gate first attempt failed (channel=$channel exit=$first_rc)" \
        "— current D1 boundary targeted refresh 후 1회 재검증" >> "$LOG"

    if [ "$D1_RECONCILE_DELAY_SECONDS" -gt 0 ]; then
        /usr/bin/sleep "$D1_RECONCILE_DELAY_SECONDS"
    fi
    if python -m data.collector_d1 \
        --refresh-current-boundary >> "$LOG" 2>&1; then
        :
    else
        refresh_rc=$?
        echo "  current D1 boundary targeted refresh failed (exit=$refresh_rc)" \
            >> "$LOG"
    fi

    if python scripts/health_check.py \
        --channel "$channel" --no-telegram >> "$LOG" 2>&1; then
        echo "  health gate recovered after bounded D1 reconcile" \
            "(channel=$channel first_exit=$first_rc refresh_exit=$refresh_rc)" \
            >> "$LOG"
        return 0
    else
        final_rc=$?
    fi
    echo "  health gate still failed after bounded D1 reconcile" \
        "(channel=$channel first_exit=$first_rc refresh_exit=$refresh_rc" \
        "final_exit=$final_rc)" >> "$LOG"
    return "$final_rc"
}

# R1은 D1만 필요하다. 느린 4h 수집/구형 distribution 예측보다 먼저 발송해
# 09:00 open 이후 불필요한 지연을 줄인다.
RECOMMEND_HEALTH_OK=0
DISTRIBUTION_HEALTH_OK=0
D1_UPDATE_OK=0
H4_UPDATE_OK=0

echo "[1/10] data update — d1" >> "$LOG"
if python -m data.collector_d1 --update >> "$LOG" 2>&1; then
    D1_UPDATE_OK=1
else
    record_critical_failure "$?" "D1 universe update"
    echo "  R1/R2/A1/PUMP send·record skipped (partial/failed D1 update)" >> "$LOG"
fi

# D1-only hard gate. 실패하면 stale R1/R2/A1/PUMP-v1 생성을 건너뛰되
# 4h 유지보수와 pump-v2 terminal 증거 검증은 계속하고 nonzero를 전파한다.
echo "[2/10] health_check gate (recommend: d1 only)" >> "$LOG"
if [ "$D1_UPDATE_OK" -eq 1 ]; then
    if run_health_with_d1_reconcile recommend; then
        RECOMMEND_HEALTH_OK=1
    else
        record_critical_failure "$?" "recommend D1 health gate"
        echo "  R1/R2/A1/PUMP send·record skipped (stale D1)" >> "$LOG"
    fi
fi

# R1 risk-reward 레이더 — 텔레그램 발송 재개 (2026-07-18 사용자 지시,
# DECISIONS #2 부분 개정). pump-v2의 2026-08-05 terminal KILL과 독립적으로
# 계속 운영하며, 성공 receipt가 있어야 ledger를 기록한다.
echo "[3/10] recommend_send + recommend_today (R1 open — 발송 재개 07-18)" >> "$LOG"
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
echo "[4/10] data update — 4h all" >> "$LOG"
if python -m data.collector_4h --all --days 2 >> "$LOG" 2>&1; then
    H4_UPDATE_OK=1
else
    record_critical_failure "$?" "4h universe update"
fi

echo "[5/10] health_check gate (distribution: d1 + 4h)" >> "$LOG"
if [ "$D1_UPDATE_OK" -eq 1 ] && [ "$H4_UPDATE_OK" -eq 1 ]; then
    if run_health_with_d1_reconcile distribution; then
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
echo "[6/10] predict_today_distribution (record only — no send flags)" >> "$LOG"
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
echo "[7/10] recommend_today --ranking R2 (record-only)" >> "$LOG"
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
echo "[8/10] recommend_today --ranking A1 (record-only)" >> "$LOG"
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
echo "[9/10] pump_detector_today (PUMP hunter ledger, record-only)" >> "$LOG"
if [ "$RECOMMEND_HEALTH_OK" -eq 1 ]; then
    if python scripts/pump_detector_today.py >> "$LOG" 2>&1; then
        :
    else
        record_critical_failure "$?" "PUMP hunter record-only ledger"
    fi
fi

# pump-v2는 2026-08-05 터미널 KILL로 기능 은퇴했다. v2 전용 Binance
# 수집은 중단한다. runner는 exact state+anchor를 검증한 뒤 scoring·Telegram·
# decision/receipt/ledger 전에 rc 0 no-op으로 끝나며, 증거 손상만 fail-loud다.
echo "[10/10] pump-v2 terminal no-op verification" >> "$LOG"
if python scripts/pump_detector_v2_today.py --send-telegram >> "$LOG" 2>&1; then
    :
else
    record_critical_failure "$?" "pump-v2 terminal verdict validation"
fi

# 1h/15m incremental update 는 별도 research cron 으로 분리 (Phase B/C 용, 운영 critical X)

echo "[done] $(date +%H:%M:%S) exit=$EXIT" >> "$LOG"
echo "" >> "$LOG"
exit "$EXIT"
