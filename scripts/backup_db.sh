#!/usr/bin/env bash
# Daily sqlite backup — /mnt/20t/prelude/data/*.db → backup 디스크.
# Phase X+3 (P0) — DB 백업 0 위험 fix.
#
# 흐름:
#   1. sqlite3 .backup (atomic, lock 안 잡고 안전 복사)
#   2. PRAGMA integrity_check (백업 파일 검증)
#   3. 14일 보관 (그 이상 삭제)
#   4. 실패 시 텔레그램 alert
#
# 운영: prelude-backup.timer (매일 04:00 KST).

set -uo pipefail

cd "$(dirname "$0")/.."
PROJ_ROOT="$(pwd)"
DATA_DIR="$PROJ_ROOT/data"
BACKUP_DIR="/home/soccz/22tb/backup/prelude_db"
LOG="$PROJ_ROOT/output/cron_backup.log"
DATE=$(date +%Y%m%d)

if [ -d "$PROJ_ROOT/venv" ]; then
    source "$PROJ_ROOT/venv/bin/activate"
fi
if [ -f "$PROJ_ROOT/.env" ]; then
    set -a
    source "$PROJ_ROOT/.env"
    set +a
fi

mkdir -p "$BACKUP_DIR"
mkdir -p "$(dirname "$LOG")"
echo "=== prelude DB backup $(date +%Y-%m-%d\ %H:%M:%S) ===" >> "$LOG"

notify_fail() {
    cd "$PROJ_ROOT"
    python -c "from notifier.telegram import send_telegram; send_telegram('⚠️ prelude backup: $1')" >> "$LOG" 2>&1 || true
}

# 1) .backup + integrity
#    7일 이상 안 변한 DB (binance_*, upbit_1h 같은 아카이브) 는 매일 복사 생략 —
#    마지막 백업본이 이미 같은 내용. 디스크/시간 낭비 제거.
N_OK=0
N_FAIL=0
N_SKIP=0
for db in "$DATA_DIR"/*.db; do
    [ -f "$db" ] || continue
    name=$(basename "$db" .db)
    dest="$BACKUP_DIR/${name}_${DATE}.db"
    if [ -n "$(find "$db" -mtime +7 2>/dev/null)" ]; then
        last_backup=$(ls -t "$BACKUP_DIR/${name}"_*.db 2>/dev/null | head -1)
        if [ -n "$last_backup" ]; then
            echo "  skip $name (7일+ unchanged, 보관본: $(basename "$last_backup"))" >> "$LOG"
            N_SKIP=$((N_SKIP + 1))
            continue
        fi
    fi
    echo "  backup $name → $dest" >> "$LOG"
    if sqlite3 "$db" ".backup '$dest'" 2>>"$LOG"; then
        result=$(sqlite3 "$dest" "PRAGMA integrity_check;" 2>>"$LOG")
        if [ "$result" = "ok" ]; then
            size=$(du -h "$dest" | cut -f1)
            echo "    ✅ $size · integrity ok" >> "$LOG"
            N_OK=$((N_OK + 1))
        else
            echo "    ❌ integrity FAIL: $result" >> "$LOG"
            N_FAIL=$((N_FAIL + 1))
            notify_fail "DB integrity fail: $name → $result"
        fi
    else
        echo "    ❌ .backup FAIL" >> "$LOG"
        N_FAIL=$((N_FAIL + 1))
        notify_fail ".backup fail: $name"
    fi
done

# 1b) ledger/상태 CSV·JSON 백업 — paper/shadow ledger 는 gitignore 라 git 에도 없음.
#     유실되면 forward 검증 (champion gate 누적) 전체 리셋이라 DB 만큼 중요.
LEDGER_BACKUP="$BACKUP_DIR/ledgers_${DATE}.tar.gz"
LEDGER_FILES=$(cd "$PROJ_ROOT" && ls output/paper_ledger*.csv output/shadow_ledger*.csv \
    output/champion_state.json output/ledger*.csv 2>/dev/null || true)
if [ -n "$LEDGER_FILES" ] && tar -czf "$LEDGER_BACKUP" -C "$PROJ_ROOT" $LEDGER_FILES 2>>"$LOG"; then
    size=$(du -h "$LEDGER_BACKUP" | cut -f1)
    n_files=$(tar -tzf "$LEDGER_BACKUP" | wc -l)
    echo "  ✅ ledger tar: $n_files files, $size" >> "$LOG"
else
    echo "  ❌ ledger tar FAIL" >> "$LOG"
    notify_fail "ledger CSV 백업 실패"
    N_FAIL=$((N_FAIL + 1))
fi

# 2) 14일 보관 정책 (DB + ledger tar 동일 적용)
find "$BACKUP_DIR" -name "*.db" -mtime +14 -delete 2>>"$LOG"
find "$BACKUP_DIR" -name "ledgers_*.tar.gz" -mtime +14 -delete 2>>"$LOG"

# 3) 결과 요약
echo "[done] $(date +%H:%M:%S) ok=$N_OK skip=$N_SKIP fail=$N_FAIL · 14일 이상 삭제" >> "$LOG"
echo "" >> "$LOG"

# 전체 실패 시 추가 alert
if [ $N_OK -eq 0 ] && [ $N_FAIL -gt 0 ]; then
    notify_fail "전체 DB 백업 실패 ($N_FAIL/$N_FAIL)"
fi

exit 0
