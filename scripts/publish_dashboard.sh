#!/usr/bin/env bash
# Dashboard publish — encrypted JSON 5종 → soccz.github.io 에 push.
#
# 흐름:
#   1. shared site repo에서는 origin URL만 읽음 (사람 WIP 무변경)
#   2. /home/soccz/22tb/tmp 전용 clone + 별도 build dir에서 JSON 생성
#   3. 정확한 5개 asset만 isolated clone에 commit + push
#   4. 실패 시 텔레그램 alert, 임시 clone은 guarded trap으로 정리
#
# 운영:
#   prelude-publish-dashboard.timer (KST 10:10) 가 호출. close cron 두 개
#   (09:30 distribution, 10:05 preopen) 끝난 후 5분 여유.
#
# 사용자 직접 실행:
#   bash scripts/publish_dashboard.sh

set -uo pipefail
export TZ=Asia/Seoul

cd "$(dirname "$0")/.."
PROJ_ROOT="$(pwd)"
SITE_ROOT="/home/soccz/22tb/soccz.github.io"
TMP_ROOT="/home/soccz/22tb/tmp"
TARGET_BRANCH="main"
DASH_REL="projects/prelude/dashboard"
DATA_REL="$DASH_REL/data"
LOG="$PROJ_ROOT/output/cron_publish.log"
ISOLATED_ROOT=""
PUBLISH_TREE=""
GENERATED_DIR=""
REMOTE_URL=""
VERIFIED_CLONE_DIR=""

mkdir -p "$(dirname "$LOG")"

# Acquire before environment parsing, build, clone, or git mutation.  A
# no-follow helper owns the descriptor and launches one recursive child with
# that descriptor inherited; an injected guard value must validate against the
# exact canonical inode before it can proceed.
LOCK_EXEC="$PROJ_ROOT/deploy/lock_exec.py"
LOCK_FILE="$PROJ_ROOT/output/.publish_dashboard.lock"
if [[ -v PRELUDE_PUBLISH_LOCK_FD ]]; then
    /usr/bin/python3 "$LOCK_EXEC" verify \
        --lock-file "$LOCK_FILE" \
        --fd "$PRELUDE_PUBLISH_LOCK_FD"
else
    exec /usr/bin/python3 "$LOCK_EXEC" run \
        --lock-file "$LOCK_FILE" \
        --fd-env PRELUDE_PUBLISH_LOCK_FD \
        --busy-exit 75 \
        --busy-message \
        "[fail] lock: another dashboard publish is active; retry required (exit=75)" \
        --busy-log "$LOG" \
        -- /bin/bash "$PROJ_ROOT/scripts/publish_dashboard.sh" "$@"
fi

if [ -d "$PROJ_ROOT/venv" ]; then
    # shellcheck disable=SC1091
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

if [ -L "$LOG" ] || { [ -e "$LOG" ] && [ ! -f "$LOG" ]; }; then
    echo "publish log must be a regular non-symlink file: $LOG" >&2
    exit 2
fi
echo "=== prelude publish dashboard $(date +%Y-%m-%d\ %H:%M:%S) ===" >> "$LOG"

notify_fail() {
    local stage="$1"
    local msg="$2"
    # systemd 실행은 OnFailure가 한 번만 알린다. 수동 실행일 때만 상세
    # stage 메시지를 직접 보내 generic+detail 중복을 피한다.
    if [ -n "${INVOCATION_ID:-}" ]; then
        echo "[alert delegated] systemd OnFailure ($stage)" >> "$LOG"
        return 0
    fi
    cd "$PROJ_ROOT"
    if PUBLISH_ALERT_STAGE="$stage" PUBLISH_ALERT_MESSAGE="$msg" python -c \
        "import os,sys; from notifier.telegram import send_telegram; msg='⚠️ prelude publish ['+os.environ['PUBLISH_ALERT_STAGE']+']: '+os.environ['PUBLISH_ALERT_MESSAGE']; sys.exit(0 if send_telegram(msg) else 1)" \
        >> "$LOG" 2>&1; then
        echo "[alert sent] publish $stage failure" >> "$LOG"
    else
        echo "[alert delivery failed] publish $stage failure" >> "$LOG"
    fi
}

fail_publish() {
    local stage="$1"
    local msg="$2"
    local rc="$3"
    echo "[fail] $stage: $msg (exit=$rc)" >> "$LOG"
    notify_fail "$stage" "$msg (exit=$rc)"
    exit "$rc"
}

cleanup_isolated_tree() {
    [ -n "$ISOLATED_ROOT" ] || return 0
    # Never let a malformed/empty variable broaden cleanup beyond the
    # dedicated prelude directory created by mktemp.
    case "$ISOLATED_ROOT" in
        "$TMP_ROOT"/prelude-dashboard-publish.*)
            /usr/bin/rm -rf -- "$ISOLATED_ROOT"
            ISOLATED_ROOT=""
            ;;
        *)
            echo "[cleanup refused] unexpected isolated path: $ISOLATED_ROOT" \
                >> "$LOG"
            ;;
    esac
}
terminate_publish() {
    local signal_name="$1"
    local exit_code="$2"
    trap - HUP INT TERM
    echo "[fail] signal: received $signal_name (exit=$exit_code)" >> "$LOG"
    exit "$exit_code"
}
trap cleanup_isolated_tree EXIT
trap 'terminate_publish HUP 129' HUP
trap 'terminate_publish INT 130' INT
trap 'terminate_publish TERM 143' TERM

preflight_dashboard_secret() {
    if python -c \
        "from scripts.build_dashboard import resolve_dashboard_passphrase; resolve_dashboard_passphrase()" \
        >> "$LOG" 2>&1; then
        return 0
    fi
    fail_publish \
        "preflight" \
        "PRELUDE_DASHBOARD_PIN missing or invalid; encrypted publish forbidden" \
        2
}

remote_url_is_allowed() {
    local remote="$1"

    # Protect every repository-taking git command from option injection and
    # reject ambiguous relative/helper-like transports. Production uses HTTPS
    # or SSH; absolute/file URLs remain available for local bare repositories.
    [[ "$remote" != -* ]] || return 1
    case "$remote" in
        /*|file:///*|https://*|ssh://*) return 0 ;;
    esac
    [[ "$remote" =~ ^[A-Za-z0-9._-]+@[A-Za-z0-9.-]+:.+$ ]]
}

assert_clone_directory() {
    local relative="$1"
    local stage="$2"
    local isolated_real
    local clone_real
    local current="$PUBLISH_TREE"
    local component
    local resolved
    local -a components=()

    VERIFIED_CLONE_DIR=""
    if [ ! -d "$ISOLATED_ROOT" ] || [ -L "$ISOLATED_ROOT" ] ||
       ! isolated_real=$(
           /usr/bin/realpath -e -- "$ISOLATED_ROOT" 2>>"$LOG"
       ); then
        fail_publish "$stage" "isolated publish root is not a real directory" 2
    fi
    if [ ! -d "$PUBLISH_TREE" ] || [ -L "$PUBLISH_TREE" ] ||
       ! clone_real=$(
           /usr/bin/realpath -e -- "$PUBLISH_TREE" 2>>"$LOG"
       ) ||
       [ "$clone_real" != "$isolated_real/site" ]; then
        fail_publish "$stage" "publish clone escaped its isolated root" 2
    fi

    IFS='/' read -r -a components <<< "$relative"
    for component in "${components[@]}"; do
        if [ -z "$component" ] || [ "$component" = "." ] ||
           [ "$component" = ".." ]; then
            fail_publish "$stage" "invalid dashboard target component" 2
        fi
        current="$current/$component"
        # -L is checked separately so dangling links also fail closed.
        if [ -L "$current" ] || [ ! -d "$current" ]; then
            fail_publish \
                "$stage" \
                "dashboard target is a symlink or non-directory: $relative" \
                2
        fi
    done
    if ! resolved=$(/usr/bin/realpath -e -- "$current" 2>>"$LOG"); then
        fail_publish "$stage" "cannot resolve dashboard target: $relative" 2
    fi
    case "$resolved" in
        "$clone_real"/*) ;;
        *)
            fail_publish \
                "$stage" \
                "dashboard target resolved outside isolated clone" \
                2
            ;;
    esac
    VERIFIED_CLONE_DIR="$resolved"
}

prepare_dashboard_data_directory() {
    local stage="$1"
    local data_path="$PUBLISH_TREE/$DATA_REL"

    # Validate every existing ancestor before mkdir. Never follow or replace a
    # remote-controlled symlink: a malformed origin must be fixed at source.
    assert_clone_directory "$DASH_REL" "$stage"
    if [ -L "$data_path" ] ||
       { [ -e "$data_path" ] && [ ! -d "$data_path" ]; }; then
        fail_publish \
            "$stage" \
            "dashboard data target is a symlink or non-directory" \
            2
    fi
    if [ ! -e "$data_path" ] &&
       ! /usr/bin/mkdir -- "$data_path"; then
        fail_publish "$stage" "cannot create isolated dashboard data dir" 2
    fi
    assert_clone_directory "$DATA_REL" "$stage"
}

prepare_isolated_site_repo() {
    local branch
    local cloned_origin
    local clone_status
    local is_worktree

    if [ ! -e "$SITE_ROOT/.git" ]; then
        fail_publish "preflight" "site repo missing: $SITE_ROOT" 2
    fi
    if ! is_worktree=$(git -C "$SITE_ROOT" rev-parse --is-inside-work-tree \
        2>>"$LOG") || [ "$is_worktree" != "true" ]; then
        fail_publish \
            "preflight" \
            "site source is not a valid git worktree: $SITE_ROOT" \
            2
    fi
    if ! REMOTE_URL=$(git -C "$SITE_ROOT" remote get-url origin 2>>"$LOG") ||
       [ -z "$REMOTE_URL" ] || [[ "$REMOTE_URL" == *$'\n'* ]] ||
       [[ "$REMOTE_URL" == *$'\r'* ]]; then
        fail_publish "preflight" "site origin remote is missing or invalid" 2
    fi
    if ! remote_url_is_allowed "$REMOTE_URL"; then
        fail_publish "preflight" "site origin remote scheme is not allowed" 2
    fi
    if ! git ls-remote --exit-code --heads -- "$REMOTE_URL" \
        "refs/heads/$TARGET_BRANCH" >/dev/null 2>>"$LOG"; then
        fail_publish \
            "preflight" \
            "origin branch unavailable: $TARGET_BRANCH" \
            2
    fi
    if ! /usr/bin/mkdir -p -- "$TMP_ROOT"; then
        fail_publish "preflight" "cannot create temp root: $TMP_ROOT" 2
    fi
    if ! ISOLATED_ROOT=$(
        /usr/bin/mktemp -d \
            "$TMP_ROOT/prelude-dashboard-publish.XXXXXXXX"
    ); then
        fail_publish "preflight" "cannot create isolated publish tree" 2
    fi
    PUBLISH_TREE="$ISOLATED_ROOT/site"
    GENERATED_DIR="$ISOLATED_ROOT/generated-data"
    if ! /usr/bin/mkdir -p -- "$GENERATED_DIR"; then
        fail_publish "preflight" "cannot create isolated build directory" 2
    fi
    if ! git clone --quiet --single-branch --branch "$TARGET_BRANCH" -- \
        "$REMOTE_URL" "$PUBLISH_TREE" >>"$LOG" 2>&1; then
        fail_publish "preflight" "isolated origin clone failed" 2
    fi
    if ! cloned_origin=$(
        git -C "$PUBLISH_TREE" remote get-url origin 2>>"$LOG"
    ) || [ "$cloned_origin" != "$REMOTE_URL" ]; then
        fail_publish "preflight" "isolated clone origin mismatch" 2
    fi
    if ! branch=$(
        git -C "$PUBLISH_TREE" symbolic-ref --short HEAD 2>>"$LOG"
    ) || [ "$branch" != "$TARGET_BRANCH" ]; then
        fail_publish "preflight" "isolated clone branch mismatch" 2
    fi
    if ! clone_status=$(
        git -C "$PUBLISH_TREE" status --porcelain 2>>"$LOG"
    ); then
        fail_publish "preflight" "cannot inspect isolated clone status" 2
    fi
    if [ -n "$clone_status" ]; then
        fail_publish "preflight" "isolated clone is unexpectedly dirty" 2
    fi
}

validate_asset_directory() {
    local asset_dir="$1"
    local rc
    local stage="$2"
    if python scripts/validate_dashboard_assets.py \
        --asset-dir "$asset_dir" >>"$LOG" 2>&1; then
        return 0
    else
        rc=$?
    fi
    fail_publish \
        "$stage" \
        "dashboard asset authentication/schema/currentness failed" \
        "$rc"
}

validate_generated_assets() {
    local asset
    local entry
    local -a expected=(
        summary.json
        history.json
        accuracy.json
        idea_validation.json
        findings.json
    )
    local -a entries=()

    for asset in "${expected[@]}"; do
        if [ ! -f "$GENERATED_DIR/$asset" ] ||
           [ -L "$GENERATED_DIR/$asset" ]; then
            fail_publish \
                "build" \
                "required generated asset missing or unsafe: $asset" \
                2
        fi
    done
    shopt -s nullglob dotglob
    entries=("$GENERATED_DIR"/*)
    shopt -u nullglob dotglob
    if [ "${#entries[@]}" -ne "${#expected[@]}" ]; then
        fail_publish \
            "build" \
            "generated asset set is not the five-file dashboard contract" \
            2
    fi
    for entry in "${entries[@]}"; do
        case "$(basename "$entry")" in
            summary.json|history.json|accuracy.json|idea_validation.json|findings.json)
                ;;
            *)
                fail_publish \
                    "build" \
                    "unexpected generated asset: $(basename "$entry")" \
                    2
                ;;
        esac
    done
}

install_generated_assets() {
    local asset
    local data_dir
    local -a expected=(
        summary.json
        history.json
        accuracy.json
        idea_validation.json
        findings.json
    )

    prepare_dashboard_data_directory "stage"
    data_dir="$VERIFIED_CLONE_DIR"
    # Revalidate immediately before each destructive/write phase. The clone is
    # disposable, but remote tree entries are still untrusted path material.
    assert_clone_directory "$DATA_REL" "stage"
    data_dir="$VERIFIED_CLONE_DIR"
    if ! /usr/bin/find "$data_dir" -mindepth 1 -maxdepth 1 \
        -exec /usr/bin/rm -rf -- {} +; then
        fail_publish "stage" "cannot clear isolated dashboard data dir" 2
    fi
    assert_clone_directory "$DATA_REL" "stage"
    data_dir="$VERIFIED_CLONE_DIR"
    for asset in "${expected[@]}"; do
        if ! /usr/bin/install -m 0644 \
            "$GENERATED_DIR/$asset" "$data_dir/$asset"; then
            fail_publish "stage" "cannot stage generated asset: $asset" 2
        fi
    done
    validate_asset_directory "$data_dir" "stage"
}

assert_staged_scope() {
    local staged
    local staged_path
    if ! staged=$(
        git -C "$PUBLISH_TREE" diff --cached --name-only 2>>"$LOG"
    ); then
        fail_publish "stage" "cannot inspect isolated staged paths" 2
    fi
    while IFS= read -r staged_path; do
        [ -z "$staged_path" ] && continue
        case "$staged_path" in
            "$DATA_REL"/*) ;;
            *)
                fail_publish \
                    "stage" \
                    "isolated commit escaped dashboard data: $staged_path" \
                    2
                ;;
        esac
    done <<< "$staged"
}

rebase_generated_commit() {
    local conflicts
    local conflict
    local data_dir="$PUBLISH_TREE/$DATA_REL"

    if git -C "$PUBLISH_TREE" rebase "origin/$TARGET_BRANCH" \
        >>"$LOG" 2>&1; then
        return 0
    fi
    conflicts=$(
        git -C "$PUBLISH_TREE" diff --name-only --diff-filter=U \
            2>>"$LOG"
    )
    if [ -z "$conflicts" ]; then
        git -C "$PUBLISH_TREE" rebase --abort >>"$LOG" 2>&1 || true
        return 1
    fi
    while IFS= read -r conflict; do
        [ -z "$conflict" ] && continue
        case "$conflict" in
            "$DATA_REL"/*) ;;
            *)
                git -C "$PUBLISH_TREE" rebase --abort \
                    >>"$LOG" 2>&1 || true
                return 1
                ;;
        esac
    done <<< "$conflicts"

    # Dashboard data is generated and wholly owned by this pipeline. Reapply
    # the already validated generation inside the disposable clone.
    prepare_dashboard_data_directory "rebase"
    data_dir="$VERIFIED_CLONE_DIR"
    assert_clone_directory "$DATA_REL" "rebase"
    data_dir="$VERIFIED_CLONE_DIR"
    /usr/bin/find "$data_dir" -mindepth 1 -maxdepth 1 \
        -exec /usr/bin/rm -rf -- {} + || return 1
    assert_clone_directory "$DATA_REL" "rebase"
    data_dir="$VERIFIED_CLONE_DIR"
    for asset in summary.json history.json accuracy.json \
        idea_validation.json findings.json; do
        /usr/bin/install -m 0644 \
            "$GENERATED_DIR/$asset" "$data_dir/$asset" || return 1
    done
    if ! python scripts/validate_dashboard_assets.py \
        --asset-dir "$data_dir" >>"$LOG" 2>&1; then
        return 1
    fi
    git -C "$PUBLISH_TREE" add -A -- "$DATA_REL" >>"$LOG" 2>&1 ||
        return 1
    assert_staged_scope
    if GIT_EDITOR=true git -C "$PUBLISH_TREE" rebase --continue \
        >>"$LOG" 2>&1; then
        return 0
    fi
    git -C "$PUBLISH_TREE" rebase --abort >>"$LOG" 2>&1 || true
    return 1
}

commit_staged_generation() {
    local stage="$1"
    local commit_rc

    if git -C "$PUBLISH_TREE" diff --cached --quiet -- "$DATA_REL"; then
        return 0
    fi
    if git -C "$PUBLISH_TREE" commit -m "$COMMIT_MSG" -- "$DATA_REL" \
        >>"$LOG" 2>&1; then
        return 0
    else
        commit_rc=$?
    fi
    fail_publish "$stage" "git commit failed" "$commit_rc"
}

restage_after_remote_update() {
    # A failed first push can race a remote update even when our original
    # generation produced no local diff.  Reapply the exact five-file
    # generation after every successful rebase, then commit any resulting
    # overwrite/deletion before retrying the push.
    install_generated_assets
    if ! git -C "$PUBLISH_TREE" add -A -- "$DATA_REL" >>"$LOG" 2>&1; then
        fail_publish "rebase" "git add failed after remote update" 2
    fi
    assert_staged_scope
    commit_staged_generation "rebase"
}

preflight_dashboard_secret
prepare_isolated_site_repo
if ! IFS= read -r PRELUDE_DASHBOARD_GENERATION_ID \
    < /proc/sys/kernel/random/uuid ||
   [[ ! "$PRELUDE_DASHBOARD_GENERATION_ID" =~ ^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$ ]]; then
    fail_publish "build" "cannot create canonical dashboard generation id" 2
fi
export PRELUDE_DASHBOARD_GENERATION_ID

# 0) policy competition audit — dashboard 가 stale summary 를 읽지 않게 publish 직전 갱신.
#    record-only 이므로 Telegram/champion state 는 건드리지 않는다.
echo "[0/4] policy_competition (dashboard audit refresh)" >> "$LOG"
if python -m ops.policy_competition >> "$LOG" 2>&1; then
    :
else
    fail_publish "policy" "policy_competition refresh failed; stale publish forbidden" "$?"
fi

# 1) idea-validation summary — bind the refreshed policy artifact and all
#    ledger/model inputs before build_dashboard consumes the summary.
echo "[1/4] idea_validation_report.py (lineage refresh)" >> "$LOG"
if python scripts/idea_validation_report.py >> "$LOG" 2>&1; then
    :
else
    fail_publish "idea_validation" "idea validation refresh failed; stale publish forbidden" "$?"
fi

# 2) build JSON
echo "[2/4] build_dashboard.py" >> "$LOG"
python scripts/build_dashboard.py --out-dir "$GENERATED_DIR" >> "$LOG" 2>&1
BUILD=$?
if [ $BUILD -ne 0 ]; then
    fail_publish "build" "build_dashboard failed" "$BUILD"
fi

# 3) findings.json — 세션 새 발견 차트 (DB 검증값, backtest 펌프 재검증).
#      일부만 새것인 dashboard를 배포하지 않도록 실패 시 전체 publish를 중단한다.
echo "[3/4] build_findings_dashboard.py" >> "$LOG"
if python scripts/build_findings_dashboard.py --out-dir "$GENERATED_DIR" >> "$LOG" 2>&1; then
    :
else
    fail_publish "findings" "findings build failed; stale publish forbidden" "$?"
fi

# 4) Copy only the validated generation into the disposable clone, then
#    commit/push from there. The shared soccz.github.io worktree stays untouched.
validate_generated_assets
echo "[3b/4] authenticate + validate generated assets" >> "$LOG"
validate_asset_directory "$GENERATED_DIR" "validate"
install_generated_assets
echo "[4/4] isolated git commit + push" >> "$LOG"
if ! git -C "$PUBLISH_TREE" add -A -- "$DATA_REL" >>"$LOG" 2>&1; then
    fail_publish "stage" "git add failed in isolated clone" 2
fi
assert_staged_scope
COMMIT_MSG="prelude dashboard: $(date +%Y-%m-%d) auto-update"
if git -C "$PUBLISH_TREE" diff --cached --quiet -- "$DATA_REL"; then
    echo "[skip] generated dashboard matches origin/$TARGET_BRANCH" >> "$LOG"
else
    commit_staged_generation "commit"
fi

# Push retry happens only inside the disposable clone. A concurrent remote
# update is fetched/rebased without touching or stashing the shared worktree.
PUSHED=0
for attempt in 1 2 3 4; do
    if git -C "$PUBLISH_TREE" push origin \
        "HEAD:refs/heads/$TARGET_BRANCH" >>"$LOG" 2>&1; then
        PUSHED=1
        break
    fi
    echo "[retry] isolated push $attempt failed — fetch/rebase" >> "$LOG"
    if git -C "$PUBLISH_TREE" fetch origin "$TARGET_BRANCH" \
        >>"$LOG" 2>&1; then
        if rebase_generated_commit; then
            restage_after_remote_update
        fi
    fi
    sleep 2
done
if [ "$PUSHED" -ne 1 ]; then
    fail_publish \
        "push" \
        "git push 4회 실패 (원격 미변경; 다음 실행에서 재생성)" \
        1
fi

echo "[done] $(date +%H:%M:%S) committed + pushed" >> "$LOG"
echo "" >> "$LOG"
exit 0
