#!/usr/bin/env bash
# Distribution paper/shadow ledger close-out entry.
#
# Runs after KST 09:00 daily candle boundary so yesterday's target day has
# final 4h bars. This keeps distribution paper ledger and shadow ledger aligned
# before the dashboard/idea-validation publish step.

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
LOG="$LOG_DIR/cron_close_$TODAY.log"
if [ -L "$LOG" ] || { [ -e "$LOG" ] && [ ! -f "$LOG" ]; }; then
    echo "close log must be a regular non-symlink file: $LOG" >&2
    exit 2
fi

echo "=== prelude distribution close KST $(date +%Y-%m-%d\ %H:%M:%S) ===" >> "$LOG"

EXIT=0
record_critical_failure() {
    local rc="$1"
    local step="$2"
    if [ "$EXIT" -eq 0 ]; then
        EXIT="$rc"
    fi
    echo "  [critical] $step failed (exit=$rc) — 가능한 후속 단계는 계속" >> "$LOG"
}

TARGET_DECISION_DATE=$(date -d yesterday +%F)
close_validated_cohort() {
    local cohort="$1"
    local ledger="$2"
    local step="$3"
    local canonical_date
    local count
    local decision_date
    local index
    local last_date=""
    local mode
    local plan_fd
    local plan_pid
    local plan_rc
    local rc
    local seen_target=0
    local sentinel="__PRELUDE_CLOSE_PLAN_V1_OK__"
    local -a plan_records=()

    coproc PRELUDE_CLOSE_PLAN {
        python -m ops.close_input_gate \
            --through-asof "$TARGET_DECISION_DATE" \
            --cohort "$cohort" \
            --output-format nul 2>>"$LOG"
    }
    plan_fd="${PRELUDE_CLOSE_PLAN[0]}"
    plan_pid="$PRELUDE_CLOSE_PLAN_PID"
    if ! mapfile -d '' -t plan_records <&"$plan_fd"; then
        plan_records=()
    fi
    if wait "$plan_pid"; then
        plan_rc=0
    else
        plan_rc=$?
    fi
    unset PRELUDE_CLOSE_PLAN PRELUDE_CLOSE_PLAN_PID
    if [ "$plan_rc" -ne 0 ]; then
        record_critical_failure "$plan_rc" \
            "$step evidence gate process failed"
        return
    fi
    count=${#plan_records[@]}
    if [ "$count" -lt 3 ] ||
       [ "${plan_records[$((count - 1))]}" != "$sentinel" ]; then
        record_critical_failure 2 \
            "$step evidence gate incomplete/empty plan"
        return
    fi
    unset "plan_records[$((count - 1))]"
    count=${#plan_records[@]}
    if [ "$count" -eq 0 ] || [ $((count % 2)) -ne 0 ]; then
        record_critical_failure 2 "$step evidence gate malformed plan"
        return
    fi

    for ((index = 0; index < count; index += 2)); do
        decision_date="${plan_records[index]}"
        mode="${plan_records[index + 1]}"
        if [[ ! "$decision_date" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] ||
           ! canonical_date=$(date -d "$decision_date" +%F 2>>"$LOG") ||
           [ "$canonical_date" != "$decision_date" ] ||
           [[ "$decision_date" > "$TARGET_DECISION_DATE" ]] ||
           { [ -n "$last_date" ] && ! [[ "$decision_date" > "$last_date" ]]; }; then
            record_critical_failure 2 \
                "$step evidence gate invalid/noncanonical date order"
            return
        fi
        case "$mode" in
            close|skip-zero-pick|skip-legacy-unverifiable|skip-no-decision|skip-terminal-kill|skip-policy-blocked) ;;
            *)
                record_critical_failure 2 \
                    "$step evidence gate invalid mode"
                return
                ;;
        esac
        if [ "$decision_date" = "$TARGET_DECISION_DATE" ]; then
            seen_target=$((seen_target + 1))
        fi
        last_date="$decision_date"
    done
    if [ "$seen_target" -ne 1 ]; then
        record_critical_failure 2 \
            "$step evidence gate target date missing/duplicated"
        return
    fi

    for ((index = 0; index < count; index += 2)); do
        decision_date="${plan_records[index]}"
        mode="${plan_records[index + 1]}"
        echo "  $step evidence gate: $mode ($decision_date)" >> "$LOG"
        if [ -z "$ledger" ]; then
            if python scripts/close_recommend_ledger.py \
                --cohort "$cohort" \
                --expected-mode "$mode" \
                --decision-date "$decision_date" >>"$LOG" 2>&1; then
                if [ "$mode" = "skip-zero-pick" ]; then
                    echo "  $step skipped — canonical healthy zero-pick evidence revalidated under lock" >> "$LOG"
                elif [ "$mode" = "skip-legacy-unverifiable" ]; then
                    echo "  $step skipped — pre-contract legacy-unverifiable revalidated under lock; never forward-valid" >> "$LOG"
                elif [ "$mode" = "skip-no-decision" ]; then
                    echo "  $step skipped — no canonical decision evidence and no ledger rows (send-day failure already alarmed)" >> "$LOG"
                elif [ "$mode" = "skip-terminal-kill" ]; then
                    echo "  $step skipped — pump-v2 immutable KILL is active (expected terminal no-op)" >> "$LOG"
                elif [ "$mode" = "skip-policy-blocked" ]; then
                    echo "  $step skipped — bounded superseded-policy gap (forward-invalid; no receipt/PnL fabricated)" >> "$LOG"
                fi
                continue
            else
                rc=$?
            fi
        else
            if python scripts/close_recommend_ledger.py \
                --ledger "$ledger" \
                --cohort "$cohort" \
                --expected-mode "$mode" \
                --decision-date "$decision_date" >>"$LOG" 2>&1; then
                if [ "$mode" = "skip-zero-pick" ]; then
                    echo "  $step skipped — canonical healthy zero-pick evidence revalidated under lock" >> "$LOG"
                elif [ "$mode" = "skip-legacy-unverifiable" ]; then
                    echo "  $step skipped — pre-contract legacy-unverifiable revalidated under lock; never forward-valid" >> "$LOG"
                elif [ "$mode" = "skip-no-decision" ]; then
                    echo "  $step skipped — no canonical decision evidence and no ledger rows (send-day failure already alarmed)" >> "$LOG"
                elif [ "$mode" = "skip-terminal-kill" ]; then
                    echo "  $step skipped — pump-v2 immutable KILL is active (expected terminal no-op)" >> "$LOG"
                elif [ "$mode" = "skip-policy-blocked" ]; then
                    echo "  $step skipped — bounded superseded-policy gap (forward-invalid; no receipt/PnL fabricated)" >> "$LOG"
                fi
                continue
            else
                rc=$?
            fi
        fi
        record_critical_failure "$rc" "$step ($decision_date)"
    done
}

echo "[1/3] data update — d1 + 4h + 15m all" >> "$LOG"
if python -m data.collector_d1 --update >> "$LOG" 2>&1; then
    :
else
    record_critical_failure "$?" "close D1 universe update"
fi
if python -m data.collector_4h --all --days 2 >> "$LOG" 2>&1; then
    :
else
    record_critical_failure "$?" "close 4h universe update"
fi
# 08:50 수집본의 마지막 08:45 봉은 진행 중일 수 있다. 09:30에 다시
# 받아야 day-D [09:00,D+1 09:00) 마지막 봉을 최종값으로 청산할 수 있다.
# --heal-days 3(초기값): 다운타임이 만든 '중간 구멍'은 --days 로 안 메워진다
# (수집기는 최상단 200봉+바닥 연장만) — 2026-08-03 부팅폭풍에서 07-31 96봉
# 창이 영구 partial 로 고착된 사고의 봉쇄. 15m 전용(4h/D1 은 페이지가 깊음).
if python -m data.collector_15m_upbit --all --heal-days 3 >> "$LOG" 2>&1; then
    :
else
    record_critical_failure "$?" "close 15m universe update"
fi

# ★ set -e 가드: close_paper_ledger 가 실패해도 아래 close_recommend_ledger(R1 SHADOW
#   실현)와 champion_selector(재선정)는 반드시 돌아야 한다 — 다음날 08:50/09:05 챔피언
#   결정에 직결. 가드가 없으면 set -e 가 여기서 스크립트를 죽여 R1 행 미실현 + 챔피언
#   stale 가 되고, close 스크립트엔 실패 알림이 없어 조용히 방치된다.
echo "[2/3] close_paper_ledger" >> "$LOG"
if python scripts/close_paper_ledger.py >> "$LOG" 2>&1; then
    :
else
    record_critical_failure "$?" "distribution paper close"
fi

# R1 SHADOW recommend ledger 실현 (전일 open 행을 -3%SL/+5%TP 15m 경로로 청산 + pump_hit).
# forward 표본 평가가능하려면 매일 청산 필수 — 별도 timer 없이 기존 close(09:30)에 fold.
echo "[2b/3] close_recommend_ledger (R1 SHADOW 실현)" >> "$LOG"
close_validated_cohort "r1-open" "" "R1 recommend close"

# R2 challenger ledger 실현 (동일 -3%SL/+5%TP 15m 경로). champion_selector 가 R1 vs R2 를
# forward CLOSED 로 비교하려면 R2 도 매일 청산돼야 함. 실패해도 R1/champion 무관(가드).
close_validated_cohort \
    "r2" \
    "output/shadow_ledger_recommend_r2.csv" \
    "R2 recommend close"

# A1 sustainability challenger ledger 실현 (동일 -3%SL/+5%TP 15m 경로). champion_selector 가
# R1 vs A1 을 forward CLOSED 로 비교하려면 A1 도 매일 청산돼야 함. 실패해도 R1/champion 무관(가드).
close_validated_cohort \
    "a1" \
    "output/shadow_ledger_recommend_sustain.csv" \
    "A1 recommend close"

# PUMP hunter rule detector ledger 실현 (동일 -3%SL/+5%TP 15m 경로 + pump20_hit).
# policy_competition 이 pump20 recall / net / downside 를 기존 모델들과 비교하려면
# 별도 shadow ledger 도 매일 CLOSED 로 전환돼야 한다. 실패해도 R1/champion 무관(가드).
close_validated_cohort \
    "pump-v1" \
    "output/shadow_ledger_pump_hunter.csv" \
    "PUMP hunter close"

# PUMP hunter v2 (Binance volsurge radar) ledger 실현 — 동일 경로 + exit lab 7 잣대.
close_validated_cohort \
    "pump-v2" \
    "output/shadow_ledger_pump_hunter_v2.csv" \
    "PUMP v2 close"

# champion/challenger 재선정 (unattended). 위 close 들이 forward CLOSED 행을 갱신한 *뒤*
# 돌려야 rolling 윈도가 최신 → 다음날 발송(08:50/09:05)이 새 champion 을 쓴다. 별도 timer 불필요.
echo "[2c/3] champion_selector (forward 갱신 후 재선정 — 다음날 발송용)" >> "$LOG"
if python -m ops.champion_selector >> "$LOG" 2>&1; then
    :
else
    record_critical_failure "$?" "champion_selector"
fi

echo "[2d/3] policy_competition (model + send-policy forward audit)" >> "$LOG"
if python -m ops.policy_competition >> "$LOG" 2>&1; then
    :
else
    record_critical_failure "$?" "policy_competition"
fi

echo "[3/4] train_recommendation_meta (shadow-gated)" >> "$LOG"
if python scripts/train_recommendation_meta.py >> "$LOG" 2>&1; then
    :
else
    record_critical_failure "$?" "recommendation meta train"
fi

echo "[4/4] idea_validation_report" >> "$LOG"
IDEA_REPORT_OK=0
if python scripts/idea_validation_report.py >> "$LOG" 2>&1; then
    IDEA_REPORT_OK=1
else
    record_critical_failure "$?" "idea validation report"
fi
if [ "$IDEA_REPORT_OK" -eq 1 ]; then
    if python scripts/build_idea_validation_html.py >> "$LOG" 2>&1; then
        :
    else
        record_critical_failure "$?" "idea validation HTML"
    fi
else
    echo "  idea validation HTML skipped (report failed; stale input forbidden)" >> "$LOG"
fi

echo "[done] $(date +%H:%M:%S) exit=$EXIT" >> "$LOG"
echo "" >> "$LOG"
exit $EXIT
