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
N_OK=0
N_FAIL=0
for db in "$DATA_DIR"/*.db; do
    [ -f "$db" ] || continue
    name=$(basename "$db" .db)
    dest="$BACKUP_DIR/${name}_${DATE}.db"
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

# 2) 14일 보관 정책
find "$BACKUP_DIR" -name "*.db" -mtime +14 -delete 2>>"$LOG"

# 3) 결과 요약
echo "[done] $(date +%H:%M:%S) ok=$N_OK fail=$N_FAIL · 14일 이상 삭제" >> "$LOG"
echo "" >> "$LOG"

# 전체 실패 시 추가 alert
if [ $N_OK -eq 0 ] && [ $N_FAIL -gt 0 ]; then
    notify_fail "전체 DB 백업 실패 ($N_FAIL/$N_FAIL)"
fi

exit 0
