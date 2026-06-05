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

preflight_site_repo() {
    if [ ! -d "$SITE_ROOT/.git" ]; then
        echo "[fail] site repo missing: $SITE_ROOT" >> "$LOG"
        notify_fail "preflight" "site repo missing: $SITE_ROOT"
        exit 2
    fi

    cd "$SITE_ROOT"

    # 진짜 병합충돌(unmerged)만 하드 실패 — 사람 개입 필요.
    UNMERGED=$(git diff --name-only --diff-filter=U)
    if [ -n "$UNMERGED" ]; then
        echo "[fail] site repo has unresolved conflicts before publish" >> "$LOG"
        echo "$UNMERGED" | sed 's/^/  /' >> "$LOG"
        notify_fail "preflight" "site repo has unresolved conflicts; publish skipped"
        exit 2
    fi

    # ★ self-heal (2026-06-05): 우리 소유(재생성 가능) data 디렉토리에 남은 dirty/untracked 를
    #   origin 상태로 되돌린다. 이전 run 이 build 후 commit 전에 죽으면 data 가 dirty 로 남아
    #   예전엔 "dirty before publish" 로 다음 run 들이 영구 차단됐다 → 이제 자가복구하고 계속.
    #   data 밖의 dirty(다른 프로젝트 WIP 등)는 우리가 안 건드린다(commit 은 data/ 만 scoped).
    git checkout -- "projects/prelude/dashboard/data" 2>/dev/null || true
    git clean -fdq "projects/prelude/dashboard/data" 2>/dev/null || true
    REMAIN=$(git status --porcelain --untracked-files=no -- "projects/prelude/dashboard/data")
    if [ -n "$REMAIN" ]; then
        echo "[warn] data dir self-heal 후에도 dirty (계속 진행):" >> "$LOG"
        echo "$REMAIN" | sed 's/^/  /' >> "$LOG"
    fi

    cd "$PROJ_ROOT"
}

preflight_site_repo

# 0) policy competition audit — dashboard 가 stale summary 를 읽지 않게 publish 직전 갱신.
#    record-only 이므로 Telegram/champion state 는 건드리지 않는다.
echo "[0/3] policy_competition (dashboard audit refresh)" >> "$LOG"
python -m ops.policy_competition >> "$LOG" 2>&1 \
    || echo "[warn] policy_competition 실패 (기존 competition summary 로 publish 계속)" >> "$LOG"

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

# push 재시도 루프 — 공유 site repo (다른 project 도 push) 의 동시 push·rejected·전송실패 견고화.
#   1) 먼저 그냥 push (fast path).
#   2) rejected 면 pull --rebase --autostash (다른 프로젝트 commit + 사람 unstaged WIP 보호).
#      rebase 충돌은 거의 우리 data 파일뿐 → 재생성본(rebase 의 --theirs) 우선으로 자동 해소,
#      실패 시 abort 후 다음 attempt. 최대 4회 → 그래도 안 되면 alert(수동).
PUSHED=0
for attempt in 1 2 3 4; do
    if git push >> "$LOG" 2>&1; then PUSHED=1; break; fi
    echo "[retry] push $attempt rejected — pull --rebase --autostash 후 재시도" >> "$LOG"
    if ! git pull --rebase --autostash origin main >> "$LOG" 2>&1; then
        # 충돌 = 우리 data 파일(재생성본). --theirs(=우리 commit) 로 keep 후 continue, 실패 시 abort.
        git checkout --theirs -- "projects/prelude/dashboard/data" 2>/dev/null \
            && git add "projects/prelude/dashboard/data" >> "$LOG" 2>&1 \
            && git rebase --continue >> "$LOG" 2>&1 \
            || git rebase --abort 2>/dev/null || true
    fi
    sleep 2
done
if [ "$PUSHED" -ne 1 ]; then
    echo "[fail] git push 4회 실패" >> "$LOG"
    notify_fail "push" "git push 4회 실패 (commit local 보존, 수동 push 필요)"
    exit 1
fi

echo "[done] $(date +%H:%M:%S) committed + pushed" >> "$LOG"
echo "" >> "$LOG"
exit 0
