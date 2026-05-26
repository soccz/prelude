#!/usr/bin/env bash
# Daily heartbeat — silent fail 방지.
# Phase X+3 (P0) — 모니터링 sliently-broken 위험 fix.
#
# 검사 항목 (이상 시 텔레그램 alert):
#   1. 어제 paper_ledger 새 row 들어왔나
#   2. DB integrity (간단 PRAGMA)
#   3. disk 사용량 90% 이상
#   4. publish.log 최근 line 정상
#
# 정상 시 — silent (스팸 방지). 이상 시만 alert.
# 운영: prelude-heartbeat.timer (매일 10:30 KST — close cron 다음).

set -uo pipefail

cd "$(dirname "$0")/.."
PROJ_ROOT="$(pwd)"
LOG="$PROJ_ROOT/output/cron_heartbeat.log"

if [ -d "$PROJ_ROOT/venv" ]; then
    source "$PROJ_ROOT/venv/bin/activate"
fi
if [ -f "$PROJ_ROOT/.env" ]; then
    set -a
    source "$PROJ_ROOT/.env"
    set +a
fi

mkdir -p "$(dirname "$LOG")"
echo "=== prelude heartbeat $(date +%Y-%m-%d\ %H:%M:%S) ===" >> "$LOG"

ALERTS=()
WARN() { ALERTS+=("$1"); echo "  ⚠️  $1" >> "$LOG"; }

# 1) paper_ledger — 어제 새 row 있나
YESTERDAY=$(date -d "yesterday" +%Y-%m-%d)
# preopen 은 2026-05-26 사용자 컨펌 DEMOTE 후 paper_ledger 추가 없음 (shadow 로만) — 정상.
for ledger in "$PROJ_ROOT/output/paper_ledger.csv" "$PROJ_ROOT/output/paper_ledger_preopen.csv"; do
    name=$(basename "$ledger")
    if [ ! -f "$ledger" ]; then
        # preopen 은 DEMOTE 직후라 파일 없을 수도 — alert X.
        if [ "$name" = "paper_ledger_preopen.csv" ]; then
            echo "  $name: DEMOTED — 파일 없음 (정상)" >> "$LOG"
        else
            WARN "$name 파일 없음"
        fi
        continue
    fi
    # preopen DEMOTED — paper_ledger 빈 거 정상. shadow_ledger 측 기록만 확인.
    if [ "$name" = "paper_ledger_preopen.csv" ]; then
        shadow="$PROJ_ROOT/output/shadow_ledger_preopen.csv"
        if [ -f "$shadow" ]; then
            shadow_n=$(grep -c "^$YESTERDAY," "$shadow" 2>/dev/null || echo 0)
            echo "  $name: DEMOTED — shadow row $shadow_n 개" >> "$LOG"
        else
            echo "  $name: DEMOTED — shadow ledger 미생성" >> "$LOG"
        fi
        continue
    fi
    n=$(grep -c "^$YESTERDAY," "$ledger" 2>/dev/null || echo 0)
    echo "  $name: 어제 ($YESTERDAY) row $n 개" >> "$LOG"
    # bear silence 라 0 정상. 7일 연속 0 이면 alert.
    if [ "$n" = "0" ]; then
        week_ago=$(date -d "7 days ago" +%Y-%m-%d)
        n_week=$(awk -F',' -v d="$week_ago" -v t="$YESTERDAY" 'NR>1 && $1>=d && $1<=t' "$ledger" | wc -l)
        echo "    7일 누적 $n_week 개" >> "$LOG"
        if [ "$n_week" = "0" ]; then
            WARN "$name 7일 연속 row 0 (bear silence 또는 system fail)"
        fi
    fi
done

# 2) DB integrity (빠른 점검 — d1 만)
DB="$PROJ_ROOT/data/upbit_d1.db"
if [ -f "$DB" ]; then
    result=$(sqlite3 "$DB" "PRAGMA integrity_check(1);" 2>>"$LOG")
    if [ "$result" != "ok" ]; then
        WARN "upbit_d1.db integrity FAIL: $result"
    else
        echo "  upbit_d1.db integrity ok" >> "$LOG"
    fi
fi

# 3) disk 사용량
USAGE=$(df /mnt/20t 2>/dev/null | awk 'NR==2 {print $5}' | tr -d '%')
echo "  /mnt/20t disk 사용 ${USAGE}%" >> "$LOG"
if [ -n "$USAGE" ] && [ "$USAGE" -ge 90 ]; then
    WARN "/mnt/20t disk ${USAGE}% (≥90%)"
fi

# 4) publish.log 최근 line
PUB_LOG="$PROJ_ROOT/output/cron_publish.log"
if [ -f "$PUB_LOG" ]; then
    if tail -20 "$PUB_LOG" | grep -q "\[fail\]"; then
        last_fail=$(tail -20 "$PUB_LOG" | grep "\[fail\]" | tail -1)
        WARN "publish 최근 fail: $last_fail"
    fi
fi

# 5) 결과 — 이상 시만 텔레그램
if [ ${#ALERTS[@]} -gt 0 ]; then
    MSG="⚠️ prelude heartbeat ($(date +%m-%d\ %H:%M))"$'\n'$'\n'
    for a in "${ALERTS[@]}"; do
        MSG="${MSG}- ${a}"$'\n'
    done
    cd "$PROJ_ROOT"
    python -c "from notifier.telegram import send_telegram; send_telegram('''$MSG''')" >> "$LOG" 2>&1 || true
    echo "[alert sent] ${#ALERTS[@]} issue" >> "$LOG"
else
    echo "[ok] all checks pass — silent" >> "$LOG"
fi
echo "" >> "$LOG"
exit 0
