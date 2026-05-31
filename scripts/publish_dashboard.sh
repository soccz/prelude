#!/usr/bin/env bash
# Dashboard publish — build JSON 3종 → soccz.github.io 에 push.
#
# 흐름:
#   1. build_dashboard.py 실행 → projects/prelude/dashboard/data/*.json 갱신
#   2. github.io repo 에 변경 있으면 commit + push
#   3. 실패 시 텔레그램 alert
#
# 운영:
#   prelude-publish-dashboard.timer (KST 10:10) 가 호출. close cron 두 개
#   (09:30 distribution, 10:05 preopen) 끝난 후 5분 여유.
#
# 사용자 직접 실행:
#   bash scripts/publish_dashboard.sh

set -uo pipefail

cd "$(dirname "$0")/.."
PROJ_ROOT="$(pwd)"
SITE_ROOT="/home/soccz/22tb/soccz.github.io"
DASH_DIR="$SITE_ROOT/projects/prelude/dashboard"
DATA_DIR="$DASH_DIR/data"
LOG="$PROJ_ROOT/output/cron_publish.log"

if [ -d "$PROJ_ROOT/venv" ]; then
    # shellcheck disable=SC1091
    source "$PROJ_ROOT/venv/bin/activate"
fi
if [ -f "$PROJ_ROOT/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$PROJ_ROOT/.env"
    set +a
fi

mkdir -p "$(dirname "$LOG")"
echo "=== prelude publish dashboard $(date +%Y-%m-%d\ %H:%M:%S) ===" >> "$LOG"

notify_fail() {
    local stage="$1"
    local msg="$2"
    cd "$PROJ_ROOT"
    python -c "from notifier.telegram import send_telegram; send_telegram('⚠️ prelude publish [$stage]: $msg')" \
        >> "$LOG" 2>&1 || true
}

# 1) build JSON
echo "[1/3] build_dashboard.py" >> "$LOG"
python scripts/build_dashboard.py --out-dir "$DATA_DIR" >> "$LOG" 2>&1
BUILD=$?
if [ $BUILD -ne 0 ]; then
    echo "[fail] build_dashboard exit=$BUILD" >> "$LOG"
    notify_fail "build" "build_dashboard exit=$BUILD"
    exit $BUILD
fi

# 1.5) findings.json — 세션 새 발견 차트 (DB 검증값, backtest 펌프 재검증)
#      실패해도 메인 publish 는 막지 않음 (대시보드는 findings 없어도 동작).
echo "[2/3] build_findings_dashboard.py" >> "$LOG"
python scripts/build_findings_dashboard.py --out-dir "$DATA_DIR" >> "$LOG" 2>&1 \
    || echo "[warn] build_findings_dashboard 실패 (findings 차트 stale, 메인 publish 계속)" >> "$LOG"

# 2) git add + commit + push if changed
echo "[3/3] git add + push (in $SITE_ROOT)" >> "$LOG"
cd "$SITE_ROOT"
git add projects/prelude/dashboard/data >> "$LOG" 2>&1

if git diff --cached --quiet; then
    echo "[skip] no data changes — nothing to commit" >> "$LOG"
    echo "[done] $(date +%H:%M:%S)" >> "$LOG"
    exit 0
fi

COMMIT_MSG="prelude dashboard: $(date +%Y-%m-%d) auto-update"
git commit -m "$COMMIT_MSG" >> "$LOG" 2>&1
COMMIT_RC=$?
if [ $COMMIT_RC -ne 0 ]; then
    echo "[fail] git commit exit=$COMMIT_RC" >> "$LOG"
    notify_fail "commit" "git commit exit=$COMMIT_RC"
    exit $COMMIT_RC
fi

# pull --rebase before push — 같은 site repo 를 다른 project (xsec-alpha 등) 가
# 동시 publish 시 rejected 방지. local commit 은 rebase 후 origin top 에 push.
# --autostash: working tree 에 unstaged 변경 있어도 안전 (auto stash → rebase → pop).
# (2026-05-21 사고: 사용자 papers 작업 unstaged → publish rebase fail)
git pull --rebase --autostash origin main >> "$LOG" 2>&1
PULL_RC=$?
if [ $PULL_RC -ne 0 ]; then
    echo "[fail] git pull --rebase exit=$PULL_RC (충돌 가능)" >> "$LOG"
    notify_fail "rebase" "git pull --rebase exit=$PULL_RC (수동 해결 필요)"
    exit $PULL_RC
fi

git push >> "$LOG" 2>&1
PUSH_RC=$?
if [ $PUSH_RC -ne 0 ]; then
    # rebase 후에도 push 실패 시 (인증 만료 등 더 심각한 문제)
    git pull --rebase origin main >> "$LOG" 2>&1 && git push >> "$LOG" 2>&1
    PUSH_RC=$?
fi
if [ $PUSH_RC -ne 0 ]; then
    echo "[fail] git push exit=$PUSH_RC" >> "$LOG"
    notify_fail "push" "git push exit=$PUSH_RC (commit 은 local 에 남음, 수동 push 필요)"
    exit $PUSH_RC
fi

echo "[done] $(date +%H:%M:%S) committed + pushed" >> "$LOG"
echo "" >> "$LOG"
exit 0
