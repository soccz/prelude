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
export TZ=Asia/Seoul

cd "$(dirname "$0")/.."
PROJ_ROOT="$(pwd)"
LOG="$PROJ_ROOT/output/cron_heartbeat.log"

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

mkdir -p "$(dirname "$LOG")"
if [ -L "$(dirname "$LOG")" ] || [ ! -d "$(dirname "$LOG")" ]; then
    echo "heartbeat log directory must be a real directory" >&2
    exit 2
fi
if [ -L "$LOG" ] || { [ -e "$LOG" ] && [ ! -f "$LOG" ]; }; then
    echo "heartbeat log must be a regular non-symlink file: $LOG" >&2
    exit 2
fi
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
            shadow_n=""
            if shadow_n=$(grep -c "^$YESTERDAY," "$shadow" 2>>"$LOG"); then
                shadow_rc=0
            else
                shadow_rc=$?
            fi
            if [ "$shadow_rc" -le 1 ] && \
               [[ "$shadow_n" =~ ^[0-9]+$ ]]; then
                echo "  $name: DEMOTED — shadow row $shadow_n 개" >> "$LOG"
            else
                WARN "$name shadow row count probe FAIL (exit=$shadow_rc; output='${shadow_n:-empty}')"
            fi
        else
            echo "  $name: DEMOTED — shadow ledger 미생성" >> "$LOG"
        fi
        continue
    fi
    n=""
    if n=$(grep -c "^$YESTERDAY," "$ledger" 2>>"$LOG"); then
        grep_rc=0
    else
        grep_rc=$?
    fi
    # grep rc=1은 정상 no-match(count=0), rc>=2는 일부 숫자를 출력했어도 실패.
    if [ "$grep_rc" -gt 1 ] || ! [[ "$n" =~ ^[0-9]+$ ]]; then
        WARN "$name row count probe FAIL (exit=$grep_rc; output='${n:-empty}')"
        continue
    fi
    echo "  $name: 어제 ($YESTERDAY) row $n 개" >> "$LOG"
    # bear silence 라 0 정상. 7일 연속 0 이면 alert.
    if [ "$n" = "0" ]; then
        week_ago=$(date -d "7 days ago" +%Y-%m-%d)
        n_week=""
        if ! n_week=$(awk -F',' -v d="$week_ago" -v t="$YESTERDAY" \
            'NR>1 && $1>=d && $1<=t' "$ledger" | wc -l); then
            WARN "$name 7-day row count probe FAIL"
            continue
        fi
        if ! [[ "$n_week" =~ ^[0-9]+$ ]]; then
            WARN "$name 7-day row count probe invalid: '${n_week:-empty}'"
            continue
        fi
        echo "    7일 누적 $n_week 개" >> "$LOG"
        if [ "$n_week" = "0" ]; then
            WARN "$name 7일 연속 row 0 (bear silence 또는 system fail)"
        fi
    fi
done

# 2) DB integrity (빠른 점검)
for DB in "$PROJ_ROOT/data/upbit_d1.db" "$PROJ_ROOT/data/policy_competition.db"; do
    db_name=$(basename "$DB")
    if [ ! -f "$DB" ]; then
        WARN "$db_name 파일 없음"
        continue
    fi
    result=""
    if result=$(sqlite3 "$DB" "PRAGMA integrity_check(1);" 2>>"$LOG"); then
        if [ "$result" != "ok" ]; then
            WARN "$db_name integrity FAIL: ${result:-empty output}"
        else
            echo "  $db_name integrity ok" >> "$LOG"
        fi
    else
        probe_rc=$?
        WARN "$db_name integrity probe FAIL (exit=$probe_rc): ${result:-empty output}"
    fi
done

# 2b) policy competition triplet — JSON/CSV/SQLite/input provenance 가 모두 같아야 함.
POLICY_JSON="$PROJ_ROOT/output/policy_competition_summary.json"
if [ -f "$POLICY_JSON" ]; then
    if python - >>"$LOG" 2>&1 <<'PYEOF'
import sys

import pandas as pd

from ops.policy_competition import load_policy_artifact

try:
    load_policy_artifact(
        asof=pd.Timestamp.now(tz="Asia/Seoul"),
        require_exact_asof=True,
        require_current=True,
    )
except Exception as exc:
    print(
        "policy_competition provenance invalid "
        f"({type(exc).__name__}): {exc}",
        file=sys.stderr,
    )
    raise SystemExit(1)
PYEOF
    then
        echo "  policy_competition JSON/CSV/SQLite provenance ok" >> "$LOG"
    else
        probe_rc=$?
        WARN "policy_competition provenance probe FAIL (exit=$probe_rc; details logged)"
    fi
else
    WARN "policy_competition_summary.json 없음"
fi

# 2c) ledger CSV 스키마 검증 — 행수만 보면 깨진 컬럼/timestamp 를 못 잡는다.
#     pandas parse + 필수 컬럼 존재 확인. 깨졌으면 forward 검증 전체가 오염되므로 alert.
csv_result=""
if csv_result=$(python - <<'PYEOF' 2>>"$LOG"
import pandas as pd
from pathlib import Path

REQUIRED = {
    "output/paper_ledger.csv": ["date", "coin", "status"],
    "output/paper_ledger_preopen.csv": ["date", "coin", "status"],
    "output/shadow_ledger_recommend.csv": ["date", "coin", "status", "realized_pct"],
    "output/shadow_ledger_recommend_preopen.csv": ["date", "coin", "status", "realized_pct"],
    "output/shadow_ledger_recommend_r2.csv": ["date", "coin", "status", "realized_pct"],
    "output/shadow_ledger_recommend_sustain.csv": ["date", "coin", "status", "realized_pct"],
    "output/shadow_ledger_pump_hunter.csv": ["date", "coin", "status", "realized_pct"],
    "output/shadow_ledger_pump_hunter_v2.csv": ["date", "coin", "status", "realized_pct"],
}
bad = []
for path, cols in REQUIRED.items():
    p = Path(path)
    if not p.exists():
        continue  # 미생성 ledger 는 정상 (DEMOTE/신규)
    try:
        df = pd.read_csv(p)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            bad.append(f"{p.name}: 컬럼 누락 {missing}")
    except Exception as e:
        bad.append(f"{p.name}: parse fail ({type(e).__name__})")
print("; ".join(bad) if bad else "ok")
PYEOF
); then
    if [ "$csv_result" != "ok" ]; then
        WARN "ledger CSV 스키마: ${csv_result:-empty output}"
    else
        echo "  ledger CSV 스키마 ok" >> "$LOG"
    fi
else
    probe_rc=$?
    WARN "ledger CSV 스키마 probe FAIL (exit=$probe_rc): ${csv_result:-empty output}"
fi

# 2d) 단일 snapshot → delivery receipt → 전유니버스 label/evaluation 연쇄.
#     snapshot이 생긴 날에만 요구하므로 도입 첫날·정상 무신호 과거에는 오탐하지 않는다.
if python - >>"$LOG" 2>&1 <<'PYEOF'
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from notifier.delivery_receipt import read_delivery_receipt
from ops.artifact_provenance import strict_json_object
from scripts.evaluate_recommend_score_labels import _report_digest
from scripts.pump_detector_v2_today import _validate_decision_document
from signals.recommend_score_labels import load_label_artifact
from signals.recommend_snapshot import load_snapshot

root = Path("output")
now = datetime.now(ZoneInfo("Asia/Seoul"))
today = now.date().isoformat()
yesterday = (now.date() - timedelta(days=1)).isoformat()
bad = []

today_dir = root / "recommend_snapshots" / today
for name in ("preopen_r1.json", "open_r1.json"):
    path = today_dir / name
    if not path.exists():
        bad.append(f"{name}: today's snapshot missing")
        continue
    try:
        snapshot = load_snapshot(path)
        receipt = read_delivery_receipt(snapshot)
        if receipt is None:
            bad.append(f"{name}: receipt missing")
        elif not receipt["delivery_ok"]:
            bad.append(f"{name}: delivery failed")
    except Exception as exc:
        bad.append(f"{name}: snapshot/receipt invalid ({type(exc).__name__})")

v2_decision_path = root / "pump_v2_decisions" / f"{today}.json"
if not v2_decision_path.exists():
    bad.append("pump_v2: today's decision manifest missing")
else:
    try:
        v2_payload = strict_json_object(v2_decision_path)
        v2_decision = v2_payload.get("decision")
        if not isinstance(v2_decision, dict):
            raise ValueError("decision payload missing")
        _validate_decision_document(
            v2_payload,
            v2_decision,
            v2_decision_path,
        )
    except Exception as exc:
        bad.append(f"pump_v2: decision manifest invalid ({type(exc).__name__})")

yesterday_dir = root / "recommend_snapshots" / yesterday
yesterday_snapshots = sorted(
    path for path in yesterday_dir.glob("*.json")
    if ".limit" not in path.name
)
for snapshot_path in yesterday_snapshots:
    label_path = (
        root / "recommend_score_labels" / yesterday / snapshot_path.name
    )
    if not label_path.exists():
        bad.append(f"{snapshot_path.name}: label missing")
        continue
    try:
        artifact = load_label_artifact(label_path)
        if artifact.get("artifact_status") != "complete":
            bad.append(
                f"{snapshot_path.name}: label {artifact.get('artifact_status')}"
            )
    except Exception as exc:
        bad.append(f"{snapshot_path.name}: label invalid ({type(exc).__name__})")

if yesterday_snapshots:
    evaluation_path = root / "recommend_score_label_evaluation.json"
    try:
        report = strict_json_object(evaluation_path)
        if report.get("report_payload_sha256") != _report_digest(report):
            bad.append("score evaluation checksum mismatch")
        generated = datetime.fromisoformat(report["generated_at"]).astimezone(
            ZoneInfo("Asia/Seoul")
        )
        if generated.date() != now.date():
            bad.append("score evaluation not refreshed today")
        used = {
            (str(item.get("asof")), str(item.get("slot")))
            for item in (report.get("artifacts", {}).get("used") or [])
        }
        for snapshot_path in yesterday_snapshots:
            slot = snapshot_path.name.split("_", 1)[0]
            if (yesterday, slot) not in used:
                bad.append(
                    f"{snapshot_path.name}: score evaluation did not use artifact"
                )
    except Exception as exc:
        bad.append(f"score evaluation invalid ({type(exc).__name__})")

if bad:
    print("; ".join(bad), file=sys.stderr)
    raise SystemExit(1)
PYEOF
then
    echo "  recommend snapshot chain ok" >> "$LOG"
else
    probe_rc=$?
    WARN "recommend snapshot chain probe FAIL (exit=$probe_rc; details logged)"
fi

# 3) disk 사용량 — 검사 자체의 실패/비정상 출력도 silent skip하지 않는다.
if USAGE=$(df /home/soccz/22tb 2>>"$LOG" | awk 'NR==2 {print $5}' | tr -d '%'); then
    if [[ "$USAGE" =~ ^[0-9]+$ ]]; then
        echo "  /home/soccz/22tb disk 사용 ${USAGE}%" >> "$LOG"
        if [ "$USAGE" -ge 90 ]; then
            WARN "/home/soccz/22tb disk ${USAGE}% (≥90%)"
        fi
    else
        WARN "disk usage 검사 비정상 출력: ${USAGE:-empty}"
    fi
else
    WARN "disk usage 검사 실행 실패"
fi

# 3a) 오늘 04:00 evidence backup의 terminal manifest와 실제 archive/checksum
#     SHA-256 결합을 검증한다. 어제 산출물만 남은 경우 오늘 백업 성공으로
#     오인하지 않는다.
BACKUP_DIR="${PRELUDE_BACKUP_DIR:-/home/soccz/22tb/backup/prelude_db}"
BACKUP_DATE=$(date +%Y%m%d)
# backup 자체 3000s 상한 + Persistent timer 지터(최대 수분)를 흡수하고,
# heartbeat unit 3900s 안에는 진단/경보 시간을 남긴다.
BACKUP_WAIT_SECONDS="${PRELUDE_BACKUP_WAIT_SECONDS:-3300}"
if [ -d "$BACKUP_DIR" ]; then
    if python -m ops.backup_manifest \
        --backup-dir "$BACKUP_DIR" \
        --date "$BACKUP_DATE" \
        --wait-seconds "$BACKUP_WAIT_SECONDS" \
        --poll-seconds 10 >>"$LOG" 2>&1; then
        echo "  backup manifest/archive/checksum provenance ok" >> "$LOG"
    else
        backup_probe_rc=$?
        WARN "backup provenance FAIL (date=$BACKUP_DATE exit=$backup_probe_rc; details logged)"
    fi
else
    WARN "backup 디렉토리 없음: $BACKUP_DIR"
fi

# 3b) close no-decision markers — 무결정일(발송 파이프 사망일) 가시화.
#     무추천일(zero-pick)과 달리 표본 자체가 사라진 날이므로 커버리지 분모
#     보정 근거로 최근 7일 개수를 상시 기록하고, 최근 2일 마커는 경고 승격.
ND_DIR="$PROJ_ROOT/output/close_no_decision"
ND_FILES=0
ND_DAYS=0
ND_FRESH=""
ND_PROBE_FAILED=0
if [ -d "$ND_DIR" ]; then
    for day_offset in 0 1 2 3 4 5 6; do
        nd_date=$(date -d "-${day_offset} day" +%F)
        nd_matches=""
        if nd_matches=$(find "$ND_DIR" -maxdepth 2 \
            -name "${nd_date}.json" 2>>"$LOG" | wc -l); then
            nd_probe_rc=0
        else
            nd_probe_rc=$?
        fi
        if [ "$nd_probe_rc" -ne 0 ] || \
           ! [[ "$nd_matches" =~ ^[0-9]+$ ]]; then
            WARN "close no-decision probe FAIL (date=$nd_date exit=$nd_probe_rc; output='${nd_matches:-empty}')"
            ND_PROBE_FAILED=1
            break
        fi
        ND_FILES=$((ND_FILES + nd_matches))
        if [ "$nd_matches" -gt 0 ]; then
            # 분모 보정은 "일수" 기준 — 하루에 여러 cohort 가 죽어도 1일.
            ND_DAYS=$((ND_DAYS + 1))
        fi
        if [ "$day_offset" -le 1 ] && [ "$nd_matches" -gt 0 ]; then
            ND_FRESH="$ND_FRESH ${nd_date}(cohorts=${nd_matches})"
        fi
    done
    if [ "$ND_PROBE_FAILED" -eq 0 ]; then
        echo "  close no-decision (7d): days=$ND_DAYS cohort-files=$ND_FILES" >> "$LOG"
        if [ -n "$ND_FRESH" ]; then
            WARN "close no-decision day 감지:$ND_FRESH — 발송 파이프 사망일(무추천일 아님), 커버리지 분모 보정 필요"
        fi
    fi
else
    echo "  close no-decision (7d): days=0 (marker dir 없음)" >> "$LOG"
fi

# 4) publish.log — 예정 시각 뒤에는 오늘 최신 세션의 명시적 성공 marker를 요구한다.
#    파일 없음/어제 성공/오늘 시작만 하고 중단/오늘 fail 모두 fail-closed. 예정 시각
#    전 수동 heartbeat만 예외로 두어 아직 실행되지 않은 publish를 오탐하지 않는다.
PUBLISH_SCHEDULE_HHMM=1010
PUB_LOG="$PROJ_ROOT/output/cron_publish.log"
NOW_HHMM=$(date +%H%M)
TODAY_ISO=$(date +%F)
if [[ "$NOW_HHMM" =~ ^[0-9]{4}$ ]] && \
   [ "$NOW_HHMM" -ge "$PUBLISH_SCHEDULE_HHMM" ]; then
    if [ ! -f "$PUB_LOG" ]; then
        WARN "publish 예정 시각 이후 로그 없음: $PUB_LOG"
    else
        LAST_SESSION=$(awk \
            '/=== prelude publish dashboard/{buf=""} {buf=buf $0 "\n"} END{printf "%s", buf}' \
            "$PUB_LOG")
        if ! echo "$LAST_SESSION" | grep -Fq \
            "=== prelude publish dashboard $TODAY_ISO "; then
            WARN "publish 오늘 세션 없음 또는 최신 세션 날짜 불일치"
        elif echo "$LAST_SESSION" | grep -q "\[fail\]"; then
            last_fail=$(echo "$LAST_SESSION" | grep "\[fail\]" | tail -1)
            WARN "publish 최근 fail: $last_fail"
        elif ! echo "$LAST_SESSION" | grep -Eq \
            '^\[done\] [0-9]{2}:[0-9]{2}:[0-9]{2} committed \+ pushed$'; then
            WARN "publish 오늘 최신 세션 성공 완료 marker 없음"
        else
            echo "  publish 오늘 최신 세션 완료 ok" >> "$LOG"
        fi
    fi
elif [[ "$NOW_HHMM" =~ ^[0-9]{4}$ ]]; then
    echo "  publish 예정 시각 전 — 완료 marker 검사 보류" >> "$LOG"
else
    WARN "publish schedule 검사 시각 비정상: ${NOW_HHMM:-empty}"
fi

# 4.5) v2 시한부 판정 일일 채점 (2026-07-12 신설 — 사전등록 09-01 판정·조기킬 무인 감시).
#      exit 21 = 조기킬(누적 mean<0) 발동 → 경고 합류. 그 외엔 로그에 한 줄만 (silent 원칙 유지).
V2_LINE=$(python scripts/v2_scoreboard.py 2>>"$LOG")
V2_RC=$?
echo "$V2_LINE" >> "$LOG"
if [ "$V2_RC" -eq 21 ]; then
    WARN "v2 조기 KILL 조건 발동: $V2_LINE"
elif [ "$V2_RC" -ne 0 ]; then
    WARN "v2 scoreboard 실행 실패 (exit=$V2_RC): $V2_LINE"
fi

# 5) 결과 — 이상 시만 텔레그램
EXIT=0
if [ ${#ALERTS[@]} -gt 0 ]; then
    MSG="⚠️ prelude heartbeat ($(date +%m-%d\ %H:%M))"$'\n'$'\n'
    for a in "${ALERTS[@]}"; do
        MSG="${MSG}- ${a}"$'\n'
    done
    # Telegram은 앞뒤 공백/개행을 strip해 에코하므로 trailing newline을 남기면
    # 에코 검증이 ambiguous로 오분류된다(2026-07-27 거짓 FAIL). 방어적 제거.
    MSG="${MSG%$'\n'}"
    cd "$PROJ_ROOT"
    if HEARTBEAT_MESSAGE="$MSG" python -c \
        "import os,sys; from notifier.telegram import send_telegram; sys.exit(0 if send_telegram(os.environ['HEARTBEAT_MESSAGE']) else 1)" \
        >> "$LOG" 2>&1; then
        echo "[alert sent] ${#ALERTS[@]} issue" >> "$LOG"
    else
        DELIVERY_RC=$?
        echo "[critical] heartbeat alert delivery FAIL (exit=$DELIVERY_RC)" >> "$LOG"
        # 상세 alert를 전달하지 못한 경우에만 systemd OnFailure가 generic
        # fallback alert를 시도한다. 전달 성공 뒤 exit 1이면 같은 이상을 2회 보낸다.
        EXIT=1
    fi
else
    echo "[ok] all checks pass — silent" >> "$LOG"
fi
echo "" >> "$LOG"
exit "$EXIT"
