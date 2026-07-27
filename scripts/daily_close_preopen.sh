#!/usr/bin/env bash
# Pre-open paper ledger close-out entry.
#
# The 08:50 pre-open run updates 15m data before the 09:00 candle starts.
# Close-out runs after 10:00 so first15/first30/first1h realized fields are
# all final, not partial.

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
LOG="$LOG_DIR/cron_preopen_close_$TODAY.log"
if [ -L "$LOG" ] || { [ -e "$LOG" ] && [ ! -f "$LOG" ]; }; then
    echo "pre-open close log must be a regular non-symlink file: $LOG" >&2
    exit 2
fi

echo "=== prelude pre-open close KST $(date +%Y-%m-%d\ %H:%M:%S) ===" >> "$LOG"

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
close_validated_preopen_r1() {
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
            --cohort r1-preopen \
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
            "R1 preopen recommend close evidence gate process failed"
        return
    fi
    count=${#plan_records[@]}
    if [ "$count" -lt 3 ] ||
       [ "${plan_records[$((count - 1))]}" != "$sentinel" ]; then
        record_critical_failure 2 \
            "R1 preopen recommend close evidence gate incomplete/empty plan"
        return
    fi
    unset "plan_records[$((count - 1))]"
    count=${#plan_records[@]}
    if [ "$count" -eq 0 ] || [ $((count % 2)) -ne 0 ]; then
        record_critical_failure 2 \
            "R1 preopen recommend close evidence gate malformed plan"
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
                "R1 preopen recommend close evidence gate invalid/noncanonical date order"
            return
        fi
        case "$mode" in
            close|skip-zero-pick|skip-legacy-unverifiable|skip-no-decision) ;;
            *)
                record_critical_failure 2 \
                    "R1 preopen recommend close evidence gate invalid mode"
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
            "R1 preopen recommend close evidence gate target date missing/duplicated"
        return
    fi

    for ((index = 0; index < count; index += 2)); do
        decision_date="${plan_records[index]}"
        mode="${plan_records[index + 1]}"
        echo "  R1 preopen evidence gate: $mode ($decision_date)" >> "$LOG"
        if python scripts/close_recommend_ledger.py \
            --ledger output/shadow_ledger_recommend_preopen.csv \
            --cohort r1-preopen \
            --expected-mode "$mode" \
            --decision-date "$decision_date" >>"$LOG" 2>&1; then
            if [ "$mode" = "skip-zero-pick" ]; then
                echo "  R1 preopen close skipped — canonical healthy zero-pick evidence revalidated under lock" >> "$LOG"
            elif [ "$mode" = "skip-legacy-unverifiable" ]; then
                echo "  R1 preopen close skipped — pre-contract legacy-unverifiable revalidated under lock; never forward-valid" >> "$LOG"
            elif [ "$mode" = "skip-no-decision" ]; then
                echo "  R1 preopen close skipped — no canonical decision evidence and no ledger rows (send-day failure already alarmed)" >> "$LOG"
            fi
            continue
        else
            rc=$?
        fi
        record_critical_failure "$rc" \
            "R1 preopen recommend close ($decision_date)"
    done
}

echo "[1/5] data update — 15m all 1 day" >> "$LOG"
if python -m data.collector_15m_upbit --all --days 1 >> "$LOG" 2>&1; then
    :
else
    record_critical_failure "$?" "preopen close 15m universe update"
fi

LABEL_DATE=$(date -d yesterday +%F)
echo "[2/5] full-universe score labels (through $LABEL_DATE)" >> "$LOG"
if python scripts/label_recommend_snapshots.py --through-date "$LABEL_DATE" >> "$LOG" 2>&1; then
    :
else
    rc=$?
    if [ "$rc" -eq 2 ]; then
        echo "  score label partial — 수집 보강 후 다음 실행에서 자동 재시도" >> "$LOG"
        record_critical_failure "$rc" "full-universe score labeling partial"
    else
        record_critical_failure "$rc" "full-universe score labeling"
    fi
fi

echo "[2b/5] full-universe score evaluation" >> "$LOG"
if python scripts/evaluate_recommend_score_labels.py >> "$LOG" 2>&1; then
    :
else
    record_critical_failure "$?" "full-universe score evaluation"
fi

# ★ set -e 가드: close_preopen_ledger 실패해도 아래 meta-train/idea-validation 은 계속.
echo "[3/5] close_preopen_ledger" >> "$LOG"
if python scripts/close_preopen_ledger.py >> "$LOG" 2>&1; then
    :
else
    record_critical_failure "$?" "preopen close"
fi

echo "[3b/5] close_recommend_ledger (R1 preopen 전용 원장)" >> "$LOG"
close_validated_preopen_r1

echo "[4/5] train_recommendation_meta (shadow-gated)" >> "$LOG"
if python scripts/train_recommendation_meta.py >> "$LOG" 2>&1; then
    :
else
    record_critical_failure "$?" "recommendation meta train"
fi

echo "[3b/4] policy_competition (model + send-policy forward audit)" >> "$LOG"
if python -m ops.policy_competition >> "$LOG" 2>&1; then
    :
else
    record_critical_failure "$?" "policy_competition"
fi

echo "[5/5] idea_validation_report" >> "$LOG"
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
