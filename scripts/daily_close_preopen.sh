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

# Initial 300s research budget: the 2026-09-08 evaluator took about 110.6s,
# making 120s too close to normal runtime; 300s is about 2.7x that observation.
# Bound hangs without changing core close
# or label deadlines. Failures remain nonzero/OnFailure-visible and still block
# publish; this reduces core latency, not complete research/publish isolation.
run_research_step() {
    /usr/bin/timeout --signal=TERM --kill-after=10s 300s "$@"
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
    local plan_record
    local rc
    local seen_target=0
    local sentinel="__PRELUDE_CLOSE_PLAN_V1_OK__"
    local -a plan_records=()

    # Keep a parent-owned descriptor: Bash unsets named coproc variables when
    # a fast producer exits, even before the parent can copy its fd/PID.
    exec {plan_fd}< <(
        python -m ops.close_input_gate \
            --through-asof "$TARGET_DECISION_DATE" \
            --cohort r1-preopen \
            --output-format nul 2>>"$LOG"
    )
    plan_pid=$!
    while IFS= read -r -d '' plan_record; do
        plan_records+=("$plan_record")
    done <&"$plan_fd"
    exec {plan_fd}<&-
    if wait "$plan_pid"; then
        plan_rc=0
    else
        plan_rc=$?
    fi
    if [ "$plan_rc" -ne 0 ]; then
        record_critical_failure "$plan_rc" \
            "R1 preopen recommend close evidence gate process failed"
        return
    fi
    count=${#plan_records[@]}
    if [ -n "$plan_record" ] || [ "$count" -lt 3 ] ||
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
            close|skip-zero-pick|skip-legacy-unverifiable|skip-no-decision|skip-terminal-kill|skip-policy-blocked) ;;
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
            elif [ "$mode" = "skip-policy-blocked" ]; then
                echo "  R1 preopen close skipped — bounded superseded-policy gap (forward-invalid; no receipt/PnL fabricated)" >> "$LOG"
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
# --heal-days 3(초기값): 다운타임 '중간 구멍' 치유 — --days 는 못 메운다 (2026-08-03)
if python -m data.collector_15m_upbit --all --heal-days 3 >> "$LOG" 2>&1; then
    :
else
    record_critical_failure "$?" "preopen close 15m universe update"
fi

# R1 close and full-universe labels depend on canonical evidence/15m data, not
# legacy preopen results or research reports. Complete both before that work.
echo "[3b/5] close_recommend_ledger (R1 preopen 전용 원장)" >> "$LOG"
close_validated_preopen_r1

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

# ★ set -e 가드: close_preopen_ledger 실패해도 아래 meta-train/idea-validation 은 계속.
echo "[3/5] close_preopen_ledger" >> "$LOG"
if python scripts/close_preopen_ledger.py >> "$LOG" 2>&1; then
    :
else
    record_critical_failure "$?" "preopen close"
fi

echo "[2b/5] full-universe score evaluation" >> "$LOG"
if run_research_step python scripts/evaluate_recommend_score_labels.py >> "$LOG" 2>&1; then
    :
else
    record_critical_failure "$?" "full-universe score evaluation"
fi

# Morning write-once evaluations predate the previous day's completed labels.
# Refresh a separate derived review only after core close/label work is done.
echo "[2c/5] post-label trial review (record-only; no promotion)" >> "$LOG"
if run_research_step python -m ops.recommend_trial_review --refresh >> "$LOG" 2>&1; then
    :
else
    record_critical_failure "$?" "post-label trial review"
fi

# Independent new-date L1 replay. A research failure must not suppress the
# existing recommendation/publication path; heartbeat checks the saved report.
echo "[2d/5] fixed L1 new-date validation (post-label replay only)" >> "$LOG"
if run_research_step python -m ops.recommend_book_validation --refresh >> "$LOG" 2>&1; then
    :
else
    book_validation_rc=$?
    echo "[DEGRADED] fixed L1 validation failed (exit=$book_validation_rc); core continues" >> "$LOG"
fi

echo "[4/5] train_recommendation_meta (shadow-gated)" >> "$LOG"
# Separate pre-entry L1 cohort; no frozen replay/collector or live changes.
if run_research_step python -m ops.recommend_book_forward --refresh >> "$LOG" 2>&1; then
    :
else
    echo "  [DEGRADED] L1 forward review failed (exit=$?) — core close continues" >> "$LOG"
fi

if run_research_step python scripts/train_recommendation_meta.py >> "$LOG" 2>&1; then
    :
else
    record_critical_failure "$?" "recommendation meta train"
fi

echo "[3b/4] policy_competition (model + send-policy forward audit)" >> "$LOG"
if run_research_step python -m ops.policy_competition >> "$LOG" 2>&1; then
    :
else
    record_critical_failure "$?" "policy_competition"
fi

echo "[5/5] idea_validation_report" >> "$LOG"
IDEA_REPORT_OK=0
if run_research_step python scripts/idea_validation_report.py >> "$LOG" 2>&1; then
    IDEA_REPORT_OK=1
else
    record_critical_failure "$?" "idea validation report"
fi
if [ "$IDEA_REPORT_OK" -eq 1 ]; then
    if run_research_step python scripts/build_idea_validation_html.py >> "$LOG" 2>&1; then
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
