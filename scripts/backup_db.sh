#!/usr/bin/env bash
# Daily sqlite backup — /home/soccz/22tb/prelude/data/*.db → backup 디스크.
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
umask 077
export TZ=Asia/Seoul

cd "$(dirname "$0")/.."
PROJ_ROOT="$(pwd)"
DATA_DIR="$PROJ_ROOT/data"
BACKUP_DIR="/home/soccz/22tb/backup/prelude_db"
LOG="$PROJ_ROOT/output/cron_backup.log"
DATE=$(date +%Y%m%d)

if ! mkdir -p "$BACKUP_DIR" || ! mkdir -p "$(dirname "$LOG")"; then
    echo "cannot create backup/log directories" >&2
    exit 1
fi
if [ -L "$BACKUP_DIR" ] || [ ! -d "$BACKUP_DIR" ]; then
    echo "backup destination must be a real directory: $BACKUP_DIR" >&2
    exit 1
fi
if [ -L "$(dirname "$LOG")" ] || [ ! -d "$(dirname "$LOG")" ]; then
    echo "backup log directory must be a real directory" >&2
    exit 1
fi
if [ -L "$LOG" ] || { [ -e "$LOG" ] && [ ! -f "$LOG" ]; }; then
    echo "backup log must be a regular non-symlink file: $LOG" >&2
    exit 1
fi
if ! chmod 700 "$BACKUP_DIR" || ! touch "$LOG" || ! chmod 600 "$LOG"; then
    echo "cannot enforce private backup/log permissions" >&2
    exit 1
fi

# systemd 수동 재시작/중복 scheduler가 같은 날짜 archive를 동시에 쓰지 못하게 한다.
LOCK_EXEC="$PROJ_ROOT/deploy/lock_exec.py"
LOCK_FILE="$BACKUP_DIR/.backup.lock"
if [[ -v PRELUDE_BACKUP_LOCK_FD ]]; then
    /usr/bin/python3 "$LOCK_EXEC" verify \
        --lock-file "$LOCK_FILE" \
        --fd "$PRELUDE_BACKUP_LOCK_FD"
else
    # 짧은 수동/timer 경합은 기다리되 stuck lock은 수동 실행에서도
    # 무한 대기하지 않는다. 3000s rc=124가 systemd 3600s보다 먼저
    # OnFailure 경로로 전파된다.
    exec /usr/bin/timeout --signal=TERM --kill-after=30s 3000s \
        /usr/bin/python3 "$LOCK_EXEC" run \
        --lock-file "$LOCK_FILE" \
        --fd-env PRELUDE_BACKUP_LOCK_FD \
        --busy-exit 75 \
        --wait \
        --busy-message \
        "  another backup process holds $LOCK_FILE" \
        --busy-log "$LOG" \
        -- /bin/bash "$PROJ_ROOT/scripts/backup_db.sh" "$@"
fi

if [ -d "$PROJ_ROOT/venv" ]; then
    source "$PROJ_ROOT/venv/bin/activate"
fi
if [ -e "$PROJ_ROOT/.env" ] || [ -L "$PROJ_ROOT/.env" ]; then
    # shellcheck disable=SC1091
    source "$PROJ_ROOT/deploy/load_runtime_env.sh"
    RUNTIME_PYTHON="$(command -v python)" || {
        echo "Python runtime unavailable" >&2
        exit 2
    }
    if ! load_prelude_runtime_env "$PROJ_ROOT/.env" "$RUNTIME_PYTHON"; then
        exit 2
    fi
fi
echo "=== prelude DB backup $(date +%Y-%m-%d\ %H:%M:%S) ===" >> "$LOG"

notify_fail() {
    # systemd 실행은 OnFailure가 단일 generic fallback을 담당한다. 수동 실행만
    # 여기서 상세 알림을 보내 중복 Telegram을 막는다.
    if [ -n "${INVOCATION_ID:-}" ]; then
        echo "  alert delegated to systemd OnFailure" >> "$LOG"
        return 0
    fi
    cd "$PROJ_ROOT"
    if ! BACKUP_ALERT="⚠️ prelude backup: $1" python -c \
        "import os,sys; from notifier.telegram import send_telegram; sys.exit(0 if send_telegram(os.environ['BACKUP_ALERT']) else 1)" \
        >> "$LOG" 2>&1; then
        echo "  ❌ backup alert delivery failed" >> "$LOG"
    fi
}

# 1) .backup + integrity
#    7일 이상 안 변한 DB (binance_*, upbit_1h 같은 아카이브) 는 매일 복사 생략 —
#    마지막 백업본이 이미 같은 내용. 디스크/시간 낭비 제거.
N_OK=0
N_FAIL=0
N_SKIP=0
N_DB=0
FAIL_DETAILS=""
note_failure() {
    if [ -z "$FAIL_DETAILS" ]; then
        FAIL_DETAILS="$1"
    else
        FAIL_DETAILS="$FAIL_DETAILS; $1"
    fi
}
is_sha256() {
    [ "${#1}" -eq 64 ] || return 1
    case "$1" in
        *[!0-9a-f]*)
            return 1
            ;;
    esac
}
checksum_matches() {
    local expected="$1"
    local published="$2"
    [ -e "$published" ] && [ ! -L "$published" ] && \
        [ -f "$published" ] && cmp -s "$expected" "$published" && (
            cd "$BACKUP_DIR" && \
            sha256sum -c -- "$(basename "$published")" >/dev/null 2>>"$LOG"
        )
}
published_db_generation_valid() {
    local published="$1"
    local source_name="$2"
    local published_name
    local published_stem
    local recorded_hash
    local actual_hash
    local checksum

    [ -e "$published" ] && [ ! -L "$published" ] && \
        [ -f "$published" ] || return 1
    published_name=$(basename "$published")
    case "$published_name" in
        "$source_name"_*.db) ;;
        *) return 1 ;;
    esac
    published_stem="${published_name%.db}"
    recorded_hash="${published_stem##*_}"
    is_sha256 "$recorded_hash" || return 1
    checksum="$published.sha256"
    [ -e "$checksum" ] && [ ! -L "$checksum" ] && \
        [ -f "$checksum" ] || return 1
    actual_hash=$(sha256sum -- "$published" 2>>"$LOG" | awk '{print $1}')
    [ "$actual_hash" = "$recorded_hash" ] || return 1
    printf '%s  %s\n' "$recorded_hash" "$published_name" |
        cmp -s - "$checksum" || return 1
    (
        cd "$BACKUP_DIR" && \
        sha256sum -c -- "$(basename "$checksum")" >/dev/null 2>>"$LOG"
    )
}
SQLITE_CHECK_RESULT=""
sqlite_check_exact_ok() {
    local database="$1"
    local pragma="$2"
    local check_rc
    local check_output

    check_output=""
    if check_output=$(sqlite3 "$database" "PRAGMA $pragma;" 2>>"$LOG"); then
        check_rc=0
    else
        check_rc=$?
    fi
    SQLITE_CHECK_RESULT="$check_output"
    if [ "$check_rc" -ne 0 ]; then
        SQLITE_CHECK_RESULT="sqlite exit $check_rc${check_output:+: $check_output}"
        return 1
    fi
    [ "$check_output" = "ok" ]
}
for db in "$DATA_DIR"/*.db; do
    [ -f "$db" ] || continue
    N_DB=$((N_DB + 1))
    name=$(basename "$db" .db)
    if [ -L "$db" ]; then
        echo "  ❌ source DB is a symlink: $name" >> "$LOG"
        N_FAIL=$((N_FAIL + 1))
        note_failure "source DB symlink rejected: $name"
        continue
    fi
    if [ -n "$(find "$db" -mtime +7 2>/dev/null)" ]; then
        last_backup=$(ls -t "$BACKUP_DIR/${name}"_*.db 2>/dev/null | head -1)
        # retention보다 오래된 유일본을 근거로 skip하면 아래 cleanup이 그
        # 마지막 복구본까지 지운다. 14일 안쪽이며 지금도 quick_check가
        # 통과하는 복구본이 있을 때만 생략한다(손상본 존재만으로 skip 금지).
        if [ -n "$last_backup" ] && \
           [ -n "$(find "$last_backup" -mtime -14 -print -quit 2>/dev/null)" ]; then
            last_check_ok=0
            SQLITE_CHECK_RESULT=""
            if published_db_generation_valid "$last_backup" "$name" && \
               sqlite_check_exact_ok "$last_backup" "quick_check"; then
                last_check_ok=1
                last_result="$SQLITE_CHECK_RESULT"
            else
                last_result="${SQLITE_CHECK_RESULT:-generation/hash/checksum mismatch}"
            fi
            # Recheck the content binding after SQLite opened the file so an
            # in-place replacement cannot turn a stale copy into skip proof.
            if [ "$last_check_ok" -eq 1 ] && \
               published_db_generation_valid "$last_backup" "$name"; then
                echo "  skip $name (7일+ unchanged, 검증 보관본: $(basename "$last_backup"))" >> "$LOG"
                N_SKIP=$((N_SKIP + 1))
                continue
            fi
            echo "  refresh $name (최근 보관본 generation/quick_check FAIL)" >> "$LOG"
        fi
    fi
    db_tmp="$BACKUP_DIR/.${name}_${DATE}.$$.db.partial"
    checksum_tmp="$BACKUP_DIR/.${name}_${DATE}.$$.db.sha256.partial"
    # 부팅 캐치업 등으로 수집기가 병행 쓰기 중이면 .backup 이 'database is
    # locked' 로 즉사할 수 있다(2026-08-03 실사고) — busy timeout + 재시도.
    backup_copy_ok=0
    for backup_attempt in 1 2 3; do
        if sqlite3 "$db" ".timeout 30000" ".backup '$db_tmp'" 2>>"$LOG"; then
            backup_copy_ok=1
            break
        fi
        echo "    .backup attempt $backup_attempt/3 fail — 10s 후 재시도" >> "$LOG"
        rm -f "$db_tmp"
        sleep 10
    done
    if [ "$backup_copy_ok" -eq 1 ]; then
        if sqlite_check_exact_ok "$db_tmp" "integrity_check"; then
            # 같은 날짜의 수동 재실행도 이전 복구점을 절대 덮지 않는다.
            # 완전성 검사와 file sync를 마친 bytes의 전체 SHA-256을 파일명에
            # 넣고 hard-link(O_EXCL과 같은 no-clobber 의미)로 publish한다.
            # 동일 bytes 재실행은 기존 세대를 검증/reuse하고, bytes가 바뀌면
            # 같은 날짜에 새 content-addressed 세대가 나란히 남는다.
            if ! sync -f "$db_tmp" 2>>"$LOG"; then
                rm -f "$db_tmp" "$checksum_tmp"
                echo "    ❌ durable publish FAIL (file sync)" >> "$LOG"
                N_FAIL=$((N_FAIL + 1))
                note_failure "DB durable publish fail: $name"
                continue
            fi
            db_hash=$(sha256sum "$db_tmp" 2>>"$LOG" | awk '{print $1}')
            if ! is_sha256 "$db_hash"; then
                rm -f "$db_tmp" "$checksum_tmp"
                echo "    ❌ SHA-256 계산 FAIL" >> "$LOG"
                N_FAIL=$((N_FAIL + 1))
                note_failure "DB SHA-256 fail: $name"
                continue
            fi
            dest="$BACKUP_DIR/${name}_${DATE}_${db_hash}.db"
            checksum="$dest.sha256"
            echo "  backup $name → $(basename "$dest")" >> "$LOG"

            reused=0
            dest_created=0
            if [ -e "$dest" ] || [ -L "$dest" ]; then
                if [ -L "$dest" ] || [ ! -f "$dest" ]; then
                    rm -f "$db_tmp" "$checksum_tmp"
                    echo "    ❌ content-addressed DB path is not a regular file" >> "$LOG"
                    N_FAIL=$((N_FAIL + 1))
                    note_failure "DB immutable generation path invalid: $name"
                    continue
                fi
                existing_hash=$(sha256sum "$dest" 2>>"$LOG" | awk '{print $1}')
                existing_check_ok=0
                if sqlite_check_exact_ok "$dest" "quick_check"; then
                    existing_check_ok=1
                fi
                if [ "$existing_hash" = "$db_hash" ] && \
                   [ "$existing_check_ok" -eq 1 ]; then
                    reused=1
                    rm -f "$db_tmp"
                else
                    rm -f "$db_tmp" "$checksum_tmp"
                    echo "    ❌ content-addressed DB collision/corruption" >> "$LOG"
                    N_FAIL=$((N_FAIL + 1))
                    note_failure "DB immutable generation conflict: $name"
                    continue
                fi
            elif ln "$db_tmp" "$dest" 2>>"$LOG"; then
                rm -f "$db_tmp"
                dest_created=1
            else
                # 외부 프로세스가 경합해 먼저 만들었을 수 있으므로 정확한
                # bytes+quick_check 일치만 idempotent reuse로 인정한다.
                existing_hash=""
                existing_check_ok=0
                if [ -e "$dest" ] && [ ! -L "$dest" ] && [ -f "$dest" ]; then
                    existing_hash=$(sha256sum "$dest" 2>>"$LOG" | awk '{print $1}')
                    if sqlite_check_exact_ok "$dest" "quick_check"; then
                        existing_check_ok=1
                    fi
                fi
                if [ "$existing_hash" = "$db_hash" ] && \
                   [ "$existing_check_ok" -eq 1 ]; then
                    reused=1
                    rm -f "$db_tmp"
                else
                    rm -f "$db_tmp" "$checksum_tmp"
                    echo "    ❌ immutable DB publish FAIL" >> "$LOG"
                    N_FAIL=$((N_FAIL + 1))
                    note_failure "DB immutable publish fail: $name"
                    continue
                fi
            fi

            checksum_ok=0
            checksum_created=0
            if ! printf '%s  %s\n' "$db_hash" "$(basename "$dest")" \
                    > "$checksum_tmp" || \
               ! sync -f "$checksum_tmp" 2>>"$LOG"; then
                :
            elif [ -e "$checksum" ] || [ -L "$checksum" ]; then
                if checksum_matches "$checksum_tmp" "$checksum"; then
                    checksum_ok=1
                fi
            elif ln "$checksum_tmp" "$checksum" 2>>"$LOG"; then
                rm -f "$checksum_tmp"
                checksum_ok=1
                checksum_created=1
            elif checksum_matches "$checksum_tmp" "$checksum"; then
                rm -f "$checksum_tmp"
                checksum_ok=1
            fi

            if [ "$checksum_ok" -eq 1 ] && \
               chmod 600 "$dest" "$checksum" 2>>"$LOG" && \
               sync -f "$BACKUP_DIR" 2>>"$LOG"; then
                size=$(du -h "$dest" | cut -f1)
                if [ "$reused" -eq 1 ]; then
                    echo "    ✅ $size · integrity/hash ok · immutable generation reused" >> "$LOG"
                    N_SKIP=$((N_SKIP + 1))
                else
                    echo "    ✅ $size · integrity/hash ok · immutable generation published" >> "$LOG"
                    N_OK=$((N_OK + 1))
                fi
            else
                rm -f "$checksum_tmp"
                # Remove only files created by this run. Pre-existing immutable
                # generations/checksums are never altered on a failed retry.
                if [ "$checksum_created" -eq 1 ]; then
                    rm -f "$checksum"
                fi
                if [ "$dest_created" -eq 1 ]; then
                    rm -f "$dest"
                fi
                sync -f "$BACKUP_DIR" 2>>"$LOG" || true
                echo "    ❌ checksum/directory durable publish FAIL" >> "$LOG"
                N_FAIL=$((N_FAIL + 1))
                note_failure "DB durable publish fail: $name"
            fi
        else
            result="$SQLITE_CHECK_RESULT"
            echo "    ❌ integrity FAIL: $result" >> "$LOG"
            rm -f "$db_tmp" "$checksum_tmp"
            N_FAIL=$((N_FAIL + 1))
            note_failure "DB integrity fail: $name → $result"
        fi
    else
        rm -f "$db_tmp" "$checksum_tmp"
        echo "    ❌ .backup FAIL" >> "$LOG"
        N_FAIL=$((N_FAIL + 1))
        note_failure ".backup fail: $name"
    fi
done
if [ "$N_DB" -eq 0 ]; then
    echo "  ❌ no SQLite DB files found under $DATA_DIR" >> "$LOG"
    note_failure "no SQLite DB files found"
    N_FAIL=$((N_FAIL + 1))
fi

# 1b) ledger + immutable 판단/전달/라벨/raw evidence 백업.
#     이 파일들은 gitignore이며 재생성할 수 없는 forward provenance다.
EVIDENCE_GENERATION="${DATE}_$(date +%H%M%S)_$$"
LEDGER_BACKUP="$BACKUP_DIR/ledgers_${EVIDENCE_GENERATION}.tar.gz"
CHECKSUM_BACKUP="$LEDGER_BACKUP.sha256"
EVIDENCE_MANIFEST="$BACKUP_DIR/ledgers_${DATE}.manifest"
LEDGER_TMP="$BACKUP_DIR/.ledgers_${EVIDENCE_GENERATION}.tar.gz.partial"
CHECKSUM_TMP="$BACKUP_DIR/.ledgers_${EVIDENCE_GENERATION}.tar.gz.sha256.partial"
MANIFEST_TMP="$BACKUP_DIR/.ledgers_${DATE}.${EVIDENCE_GENERATION}.manifest.partial"
BACKUP_ITEMS=()
TERMINAL_STATE="output/radar_terminal_verdict.json"
TERMINAL_ANCHOR="data/radar_terminal_verdict.anchor.json"
TERMINAL_VERDICT_PAIR="absent"
TERMINAL_VERDICT_PAIR_OK=1
shopt -s nullglob
for item in \
    "$PROJ_ROOT"/output/paper_ledger*.csv \
    "$PROJ_ROOT"/output/shadow_ledger*.csv \
    "$PROJ_ROOT"/output/ledger*.csv \
    "$PROJ_ROOT"/output/champion_state.json \
    "$PROJ_ROOT"/output/champion_state.legacy.*.json \
    "$PROJ_ROOT"/output/recommend_score_label_evaluation.json; do
    [ -e "$item" ] && BACKUP_ITEMS+=("${item#"$PROJ_ROOT"/}")
done
for item in \
    output/recommend_snapshots \
    output/recommend_receipts \
    output/pump_v1_decisions \
    output/pump_v2_receipts \
    output/pump_v2_decisions \
    output/recommend_score_labels \
    output/close_no_decision \
    data/microstructure/upbit; do
    [ -e "$PROJ_ROOT/$item" ] && BACKUP_ITEMS+=("$item")
done
# Terminal verdict state and its independent anchor form one restore unit.
# Before a verdict exists both are optional; once either exists, both must be
# captured in the same generation or the previous manifest remains current.
TERMINAL_STATE_ENTRY=0
TERMINAL_ANCHOR_ENTRY=0
if [ -e "$PROJ_ROOT/$TERMINAL_STATE" ] || \
   [ -L "$PROJ_ROOT/$TERMINAL_STATE" ]; then
    TERMINAL_STATE_ENTRY=1
fi
if [ -e "$PROJ_ROOT/$TERMINAL_ANCHOR" ] || \
   [ -L "$PROJ_ROOT/$TERMINAL_ANCHOR" ]; then
    TERMINAL_ANCHOR_ENTRY=1
fi
if [ "$TERMINAL_STATE_ENTRY" -eq 1 ] && \
   [ "$TERMINAL_ANCHOR_ENTRY" -eq 1 ]; then
    if [ -L "$PROJ_ROOT/$TERMINAL_STATE" ] || \
       [ ! -f "$PROJ_ROOT/$TERMINAL_STATE" ] || \
       [ -L "$PROJ_ROOT/$TERMINAL_ANCHOR" ] || \
       [ ! -f "$PROJ_ROOT/$TERMINAL_ANCHOR" ]; then
        TERMINAL_VERDICT_PAIR="invalid"
        TERMINAL_VERDICT_PAIR_OK=0
        echo "  ❌ terminal verdict state/anchor must be regular non-symlink files" >> "$LOG"
        note_failure "terminal verdict state/anchor files invalid"
        N_FAIL=$((N_FAIL + 1))
    else
        BACKUP_ITEMS+=("$TERMINAL_STATE" "$TERMINAL_ANCHOR")
        TERMINAL_VERDICT_PAIR="present"
    fi
elif [ "$TERMINAL_STATE_ENTRY" -eq 1 ] || \
     [ "$TERMINAL_ANCHOR_ENTRY" -eq 1 ]; then
    TERMINAL_VERDICT_PAIR="incomplete"
    TERMINAL_VERDICT_PAIR_OK=0
    echo "  ❌ terminal verdict state/anchor pair incomplete" >> "$LOG"
    note_failure "terminal verdict state/anchor pair incomplete"
    N_FAIL=$((N_FAIL + 1))
fi
shopt -u nullglob

archive_members_are_safe() {
    # Restore evidence must be self-contained: no symlink/hardlink/device
    # members and no absolute or parent-traversal archive names.
    tar -tzf "$1" 2>>"$LOG" | awk '
        /^[/]/ || $0 == ".." || /^[.][.][/]/ ||
        /[/][.][.][/]/ || /[/][.][.]$/ { bad = 1 }
        END { exit bad }
    ' && \
    tar -tvzf "$1" 2>>"$LOG" | awk '
        substr($1, 1, 1) != "-" && substr($1, 1, 1) != "d" { bad = 1 }
        END { exit bad }
    '
}

evidence_archive_semantics_valid() {
    local archive="$1"
    local expected_pair="$2"
    local python_bin

    python_bin=$(command -v python) || return 1
    "$python_bin" - "$archive" "$expected_pair" 2>>"$LOG" <<'PY'
from __future__ import annotations

import sys
import tarfile
from pathlib import Path

archive_path = Path(sys.argv[1])
expected_pair = sys.argv[2]
state_name = "output/radar_terminal_verdict.json"
anchor_name = "data/radar_terminal_verdict.anchor.json"

if expected_pair not in {"absent", "present"}:
    raise SystemExit("unsupported terminal verdict pair state")

with tarfile.open(archive_path, "r:gz") as archive:
    selected = [
        member
        for member in archive.getmembers()
        if member.name in {state_name, anchor_name}
    ]
    if expected_pair == "absent":
        if selected:
            raise SystemExit(
                "terminal verdict members exist in an absent-pair archive"
            )
        raise SystemExit(0)

    if (
        len(selected) != 2
        or {member.name for member in selected} != {state_name, anchor_name}
        or any(not member.isfile() for member in selected)
    ):
        raise SystemExit(
            "terminal verdict archive members are incomplete or ambiguous"
        )
    members = {member.name: member for member in selected}
    state_handle = archive.extractfile(members[state_name])
    anchor_handle = archive.extractfile(members[anchor_name])
    if state_handle is None or anchor_handle is None:
        raise SystemExit("terminal verdict archive members are unreadable")
    state_bytes = state_handle.read()
    anchor_bytes = anchor_handle.read()

from ops.artifact_provenance import strict_json_object_bytes
from ops.radar_verdict import _validate_payload

state = _validate_payload(
    strict_json_object_bytes(state_bytes, source=state_name)
)
anchor = _validate_payload(
    strict_json_object_bytes(anchor_bytes, source=anchor_name)
)
if state != anchor:
    raise SystemExit("terminal verdict state/anchor semantic mismatch")
PY
}

EVIDENCE_OK=0
ARCHIVE_PUBLISHED=0
ARCHIVE_CREATED=0
PAIR_PUBLISHED=0
CHECKSUM_CREATED=0
MANIFEST_PUBLISHED=0
if [ "$TERMINAL_VERDICT_PAIR_OK" -eq 1 ] && \
   [ ${#BACKUP_ITEMS[@]} -gt 0 ] && \
   tar -czf "$LEDGER_TMP" -C "$PROJ_ROOT" "${BACKUP_ITEMS[@]}" 2>>"$LOG" && \
   tar -tzf "$LEDGER_TMP" >/dev/null 2>>"$LOG" && \
   archive_members_are_safe "$LEDGER_TMP" && \
   evidence_archive_semantics_valid \
       "$LEDGER_TMP" "$TERMINAL_VERDICT_PAIR" && \
   sync -f "$LEDGER_TMP" 2>>"$LOG"; then
    archive_hash=$(sha256sum "$LEDGER_TMP" 2>>"$LOG" | awk '{print $1}')
    if is_sha256 "$archive_hash" && \
       printf '%s  %s\n' "$archive_hash" "$(basename "$LEDGER_BACKUP")" \
           > "$CHECKSUM_TMP" && \
       printf 'schema=prelude_evidence_backup.v1\n' > "$MANIFEST_TMP" && \
       printf 'date=%s\n' "$DATE" >> "$MANIFEST_TMP" && \
       printf 'generation=%s\n' "$EVIDENCE_GENERATION" >> "$MANIFEST_TMP" && \
       printf 'archive=%s\n' "$(basename "$LEDGER_BACKUP")" \
           >> "$MANIFEST_TMP" && \
       printf 'checksum=%s\n' "$(basename "$CHECKSUM_BACKUP")" \
           >> "$MANIFEST_TMP" && \
       printf 'sha256=%s\n' "$archive_hash" >> "$MANIFEST_TMP" && \
       printf 'terminal_verdict_pair=%s\n' "$TERMINAL_VERDICT_PAIR" \
           >> "$MANIFEST_TMP" && \
       sync -f "$CHECKSUM_TMP" 2>>"$LOG" && \
       sync -f "$MANIFEST_TMP" 2>>"$LOG"; then
        if [ -e "$LEDGER_BACKUP" ] || [ -L "$LEDGER_BACKUP" ]; then
            if [ -L "$LEDGER_BACKUP" ] || [ ! -f "$LEDGER_BACKUP" ]; then
                existing_hash=""
            else
                existing_hash=$(sha256sum "$LEDGER_BACKUP" 2>>"$LOG" | awk '{print $1}')
            fi
            if [ "$existing_hash" = "$archive_hash" ] && \
               tar -tzf "$LEDGER_BACKUP" >/dev/null 2>>"$LOG" && \
               archive_members_are_safe "$LEDGER_BACKUP" && \
               evidence_archive_semantics_valid \
                   "$LEDGER_BACKUP" "$TERMINAL_VERDICT_PAIR"; then
                rm -f "$LEDGER_TMP"
                ARCHIVE_PUBLISHED=1
            fi
        elif ln "$LEDGER_TMP" "$LEDGER_BACKUP" 2>>"$LOG"; then
            rm -f "$LEDGER_TMP"
            ARCHIVE_PUBLISHED=1
            ARCHIVE_CREATED=1
        elif [ -e "$LEDGER_BACKUP" ] && [ ! -L "$LEDGER_BACKUP" ] && \
             [ -f "$LEDGER_BACKUP" ]; then
            existing_hash=$(sha256sum "$LEDGER_BACKUP" 2>>"$LOG" | awk '{print $1}')
            if [ "$existing_hash" = "$archive_hash" ] && \
               tar -tzf "$LEDGER_BACKUP" >/dev/null 2>>"$LOG" && \
               archive_members_are_safe "$LEDGER_BACKUP" && \
               evidence_archive_semantics_valid \
                   "$LEDGER_BACKUP" "$TERMINAL_VERDICT_PAIR"; then
                rm -f "$LEDGER_TMP"
                ARCHIVE_PUBLISHED=1
            fi
        fi
        if [ "$ARCHIVE_PUBLISHED" -eq 1 ]; then
            if checksum_matches "$CHECKSUM_TMP" "$CHECKSUM_BACKUP"; then
                rm -f "$CHECKSUM_TMP"
                PAIR_PUBLISHED=1
            elif [ ! -e "$CHECKSUM_BACKUP" ] && \
                 ln "$CHECKSUM_TMP" "$CHECKSUM_BACKUP" 2>>"$LOG"; then
                rm -f "$CHECKSUM_TMP"
                PAIR_PUBLISHED=1
                CHECKSUM_CREATED=1
            elif checksum_matches "$CHECKSUM_TMP" "$CHECKSUM_BACKUP"; then
                rm -f "$CHECKSUM_TMP"
                PAIR_PUBLISHED=1
            fi
        fi
        # The unique archive/checksum pair is durable and independently
        # verified before the date manifest can point at it.
        if [ "$PAIR_PUBLISHED" -eq 1 ] && \
           chmod 600 "$LEDGER_BACKUP" "$CHECKSUM_BACKUP" 2>>"$LOG" && \
           sync -f "$BACKUP_DIR" 2>>"$LOG" && \
           (
               cd "$BACKUP_DIR" && \
               sha256sum -c "$(basename "$CHECKSUM_BACKUP")" >/dev/null 2>>"$LOG"
           ) && \
           evidence_archive_semantics_valid \
               "$LEDGER_BACKUP" "$TERMINAL_VERDICT_PAIR" && \
           mv "$MANIFEST_TMP" "$EVIDENCE_MANIFEST" 2>>"$LOG"; then
            MANIFEST_PUBLISHED=1
        fi
        if [ "$MANIFEST_PUBLISHED" -eq 1 ] && \
           sync -f "$BACKUP_DIR" 2>>"$LOG"; then
            EVIDENCE_OK=1
        fi
    fi
fi

if [ "$EVIDENCE_OK" -eq 1 ]; then
    size=$(du -h "$LEDGER_BACKUP" | cut -f1)
    n_files=$(tar -tzf "$LEDGER_BACKUP" | wc -l)
    echo "  ✅ evidence tar: $n_files files, $size · generation=$EVIDENCE_GENERATION · manifest-last commit" >> "$LOG"
else
    rm -f "$LEDGER_TMP" "$CHECKSUM_TMP" "$MANIFEST_TMP"
    # A failure before the manifest commit must not disturb the previously
    # committed generation. Remove only this run's unreferenced version files.
    if [ "$MANIFEST_PUBLISHED" -eq 0 ]; then
        [ "$CHECKSUM_CREATED" -eq 0 ] || rm -f "$CHECKSUM_BACKUP"
        [ "$ARCHIVE_CREATED" -eq 0 ] || rm -f "$LEDGER_BACKUP"
    fi
    echo "  ❌ evidence tar FAIL" >> "$LOG"
    note_failure "ledger/provenance 증거 백업 실패"
    N_FAIL=$((N_FAIL + 1))
fi

# 2) 14일 보관 정책 (DB + ledger tar 동일 적용)
if [ "$N_FAIL" -eq 0 ]; then
    if find "$BACKUP_DIR" \
        \( -name "*.db" -o -name "*.db.sha256" \) \
        -mtime +14 -delete 2>>"$LOG"; then
        :
    else
        echo "  ❌ old DB retention cleanup FAIL" >> "$LOG"
        note_failure "오래된 DB 백업 정리 실패"
        N_FAIL=$((N_FAIL + 1))
    fi
    if find "$BACKUP_DIR" \
        \( -name "ledgers_*.tar.gz" -o \
           -name "ledgers_*.tar.gz.sha256" -o \
           -name "ledgers_*.manifest" \) \
        -mtime +14 -delete 2>>"$LOG"; then
        :
    else
        echo "  ❌ old evidence retention cleanup FAIL" >> "$LOG"
        note_failure "오래된 ledger/provenance 백업 정리 실패"
        N_FAIL=$((N_FAIL + 1))
    fi
else
    echo "  retention skipped because this run has failures" >> "$LOG"
fi

# 3) 결과 요약
echo "[done] $(date +%H:%M:%S) ok=$N_OK skip=$N_SKIP fail=$N_FAIL · 14일 이상 삭제" >> "$LOG"
echo "" >> "$LOG"

if [ "$N_FAIL" -gt 0 ]; then
    notify_fail "fail=$N_FAIL, ok=$N_OK, skip=$N_SKIP — $FAIL_DETAILS"
    exit 1
fi
exit 0
