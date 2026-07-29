#!/usr/bin/env bash
# Install the single supported prelude scheduler: eight systemd timers.
#
# Production:
#   sudo bash deploy/install_systemd.sh
#
# Read-only validation (cron spool inspection still requires root):
#   sudo bash deploy/install_systemd.sh --check-only

set -euo pipefail
umask 077

CHECK_ONLY=0
FIXTURE_INSTALL=0
case "${1:-}" in
    "") ;;
    --check-only) CHECK_ONLY=1 ;;
    -h|--help)
        sed -n '1,9p' "$0"
        exit 0
        ;;
    *)
        echo "ERROR: unknown argument: $1" >&2
        exit 64
        ;;
esac

die() {
    echo "ERROR: $*" >&2
    exit 1
}

if [ "${PRELUDE_INSTALL_FIXTURE:-0}" = "1" ]; then
    [ "$CHECK_ONLY" -eq 0 ] ||
        die "PRELUDE_INSTALL_FIXTURE is only valid for an install test"
    [ "$EUID" -ne 0 ] ||
        die "fixture install must never run with root privileges"
    for required_override in \
        PRELUDE_INSTALL_REPO \
        PRELUDE_INSTALL_UNIT_DIR \
        PRELUDE_INSTALL_CRON_ROOT \
        PRELUDE_INSTALL_SYSTEMCTL \
        PRELUDE_INSTALL_ANALYZE \
        PRELUDE_INSTALL_ENV_FILE; do
        [ -n "${!required_override:-}" ] ||
            die "fixture install requires $required_override"
    done
    FIXTURE_INSTALL=1
fi

if [ "$CHECK_ONLY" -eq 1 ] || [ "$FIXTURE_INSTALL" -eq 1 ]; then
    # Alternate roots/binaries make the preflight executable in an isolated
    # test fixture.  A privileged real install always ignores these overrides.
    REPO="${PRELUDE_INSTALL_REPO:-/home/soccz/22tb/prelude}"
    UNIT_DIR="${PRELUDE_INSTALL_UNIT_DIR:-/etc/systemd/system}"
    CRON_ROOT="${PRELUDE_INSTALL_CRON_ROOT:-/}"
    SYSTEMCTL_BIN="${PRELUDE_INSTALL_SYSTEMCTL:-/usr/bin/systemctl}"
    SYSTEMD_ANALYZE_BIN="${PRELUDE_INSTALL_ANALYZE:-/usr/bin/systemd-analyze}"
    ENV_FILE="${PRELUDE_INSTALL_ENV_FILE:-$REPO/.env}"
else
    REPO="/home/soccz/22tb/prelude"
    UNIT_DIR="/etc/systemd/system"
    CRON_ROOT="/"
    SYSTEMCTL_BIN="/usr/bin/systemctl"
    SYSTEMD_ANALYZE_BIN="/usr/bin/systemd-analyze"
    ENV_FILE="$REPO/.env"
fi
UNIT_USER="soccz"

if [ "$FIXTURE_INSTALL" -eq 1 ]; then
    case "$UNIT_DIR/" in
        "${CRON_ROOT%/}/"*) ;;
        *) die "fixture unit directory must be contained by its fixture root" ;;
    esac
    for fixture_path in "$SYSTEMCTL_BIN" "$SYSTEMD_ANALYZE_BIN" "$ENV_FILE"; do
        case "$fixture_path" in
            "${CRON_ROOT%/}/"*) ;;
            *) die "fixture dependency escapes its fixture root: $fixture_path" ;;
        esac
    done
fi

SERVICE_UNITS=(
    prelude-distribution.service
    prelude-close.service
    prelude-preopen.service
    prelude-preopen-close.service
    prelude-publish-dashboard.service
    prelude-backup.service
    prelude-heartbeat.service
    prelude-selftest.service
    prelude-failure-alert@.service
)
TIMER_UNITS=(
    prelude-distribution.timer
    prelude-close.timer
    prelude-preopen.timer
    prelude-preopen-close.timer
    prelude-publish-dashboard.timer
    prelude-backup.timer
    prelude-heartbeat.timer
    prelude-selftest.timer
)
ALL_UNITS=("${SERVICE_UNITS[@]}" "${TIMER_UNITS[@]}")

if [ "$EUID" -ne 0 ] && [ "$CRON_ROOT" = "/" ]; then
    die "root is required to inspect every user's cron spool (use sudo)"
fi

root_path() {
    local absolute_path="$1"
    if [ "$CRON_ROOT" = "/" ]; then
        printf '%s\n' "$absolute_path"
    else
        printf '%s%s\n' "${CRON_ROOT%/}" "$absolute_path"
    fi
}

is_expected_timer() {
    local candidate="$1"
    local expected
    for expected in "${TIMER_UNITS[@]}"; do
        if [ "$candidate" = "$expected" ]; then
            return 0
        fi
    done
    return 1
}

is_expected_service() {
    local candidate="$1"
    local expected
    for expected in "${SERVICE_UNITS[@]}"; do
        if [ "$candidate" = "$expected" ]; then
            return 0
        fi
    done
    return 1
}

scan_cron_file() {
    local cron_file="$1"
    local matches

    [ -r "$cron_file" ] || die "cannot read cron source; refusing fail-open: $cron_file"
    if ! matches=$(
        /usr/bin/awk -v repo="$REPO" '
            /^[[:space:]]*($|#)/ { next }
            index($0, repo) ||
            /prelude/ ||
            /daily_run_(preopen|distribution)\.sh/ ||
            /daily_close_(preopen|distribution)\.sh/ ||
            /publish_dashboard\.sh/ ||
            /backup_db\.sh/ ||
            /heartbeat\.sh/ { print FNR ": " $0 }
        ' "$cron_file"
    ); then
        die "cannot inspect cron source; refusing fail-open: $cron_file"
    fi
    if [ -n "$matches" ]; then
        echo "ERROR: active prelude cron found in $cron_file:" >&2
        echo "$matches" >&2
        echo "Remove or comment these entries before enabling systemd timers." >&2
        exit 1
    fi
}

validate_env_contract() {
    local env_mode
    local env_owner
    local env_uid
    local -a parser_command=(
        /usr/bin/python3
        -I
        "$REPO/ops/runtime_env.py"
        "$ENV_FILE"
        --require TELEGRAM_BOT_TOKEN
        --require TELEGRAM_CHAT_ID
        --require PRELUDE_DASHBOARD_PIN
    )

    [ -e "$ENV_FILE" ] || die "runtime environment file missing: $ENV_FILE"
    [ ! -L "$ENV_FILE" ] || die "runtime environment must not be a symlink"
    [ -f "$ENV_FILE" ] || die "runtime environment is not a regular file"
    env_owner=$(/usr/bin/stat -c '%U' "$ENV_FILE")
    [ "$env_owner" = "$UNIT_USER" ] ||
        die ".env owner must be $UNIT_USER (found $env_owner)"
    env_mode=$(/usr/bin/stat -c '%a' "$ENV_FILE")
    case "$env_mode" in
        400|600) ;;
        *) die ".env mode must be 0400 or 0600 (found $env_mode)" ;;
    esac
    env_uid=$(/usr/bin/stat -c '%u' "$ENV_FILE")
    [ -x /usr/bin/python3 ] || die "system Python is unavailable"

    # REPO and its venv are writable by UNIT_USER.  During a privileged
    # install, drop credentials before Python opens either repository code or
    # configuration; isolated mode also ignores PYTHON* injection.
    if [ "$EUID" -eq 0 ]; then
        [ -x /usr/sbin/runuser ] || die "runuser is unavailable"
        /usr/sbin/runuser --user "$UNIT_USER" -- \
            "${parser_command[@]}" ||
            die "runtime .env contract failed"
    else
        [ "$EUID" -eq "$env_uid" ] ||
            die "non-root preflight must run as .env owner $UNIT_USER"
        "${parser_command[@]}" ||
            die "runtime .env contract failed"
    fi
}

validate_repo_contract() {
    local repo_owner
    local required_binary
    local required_script
    local unit
    local verify_paths=()

    [ -d "$REPO" ] || die "repository missing: $REPO"
    repo_owner=$(/usr/bin/stat -c '%U' "$REPO")
    [ "$repo_owner" = "$UNIT_USER" ] ||
        die "unit User=$UNIT_USER but repository owner is $repo_owner"
    [ -x "$REPO/venv/bin/python" ] || die "venv Python is not executable"
    validate_env_contract
    for required_binary in \
        /bin/bash \
        /usr/bin/awk \
        /usr/bin/basename \
        /usr/bin/chmod \
        /usr/bin/cmp \
        /usr/bin/cp \
        /usr/bin/cut \
        /usr/bin/date \
        /usr/bin/df \
        /usr/bin/dirname \
        /usr/bin/du \
        /usr/bin/find \
        /usr/bin/flock \
        /usr/bin/git \
        /usr/bin/grep \
        /usr/bin/gzip \
        /usr/bin/head \
        /usr/bin/install \
        /usr/bin/ln \
        /usr/bin/ls \
        /usr/bin/mkdir \
        /usr/bin/mktemp \
        /usr/bin/mv \
        /usr/bin/python3 \
        /usr/bin/realpath \
        /usr/bin/rm \
        /usr/bin/sed \
        /usr/bin/sha256sum \
        /usr/bin/sleep \
        /usr/bin/sqlite3 \
        /usr/bin/stat \
        /usr/bin/sync \
        /usr/bin/tail \
        /usr/bin/tar \
        /usr/bin/touch; do
        [ -x "$required_binary" ] ||
            die "required runtime binary missing: $required_binary"
    done

    for required_script in \
        deploy/lock_exec.py \
        deploy/load_runtime_env.sh \
        ops/runtime_env.py \
        scripts/daily_run_distribution.sh \
        scripts/daily_close_distribution.sh \
        scripts/daily_run_preopen.sh \
        scripts/daily_close_preopen.sh \
        scripts/publish_dashboard.sh \
        scripts/validate_dashboard_assets.py \
        scripts/backup_db.sh \
        scripts/heartbeat.sh \
        scripts/systemd_failure_alert.py; do
        [ -e "$REPO/$required_script" ] ||
            die "required runtime missing: $required_script"
        [ ! -L "$REPO/$required_script" ] ||
            die "required runtime must not be a symlink: $required_script"
        [ -f "$REPO/$required_script" ] ||
            die "required runtime is not a regular file: $required_script"
        [ -r "$REPO/$required_script" ] ||
            die "required runtime is not readable: $required_script"
    done
    [ -e "$REPO/deploy/run_pipeline_stage.sh" ] &&
        [ ! -L "$REPO/deploy/run_pipeline_stage.sh" ] &&
        [ -f "$REPO/deploy/run_pipeline_stage.sh" ] &&
        [ -x "$REPO/deploy/run_pipeline_stage.sh" ] ||
        die "pipeline wrapper is not executable"

    for unit in "${ALL_UNITS[@]}"; do
        [ -e "$REPO/deploy/$unit" ] ||
            die "unit source missing: deploy/$unit"
        [ ! -L "$REPO/deploy/$unit" ] ||
            die "unit source must not be a symlink: deploy/$unit"
        [ -f "$REPO/deploy/$unit" ] ||
            die "unit source is not a regular file: deploy/$unit"
        [ -r "$REPO/deploy/$unit" ] ||
            die "unit source is not readable: deploy/$unit"
        verify_paths+=("$REPO/deploy/$unit")
    done
    [ -x "$SYSTEMD_ANALYZE_BIN" ] || die "systemd-analyze unavailable: $SYSTEMD_ANALYZE_BIN"
    "$SYSTEMD_ANALYZE_BIN" verify "${verify_paths[@]}" ||
        die "systemd-analyze rejected one or more source units"
}

validate_installed_contract() {
    local fragment_path
    local installed_path
    local installed_mode
    local installed_owner
    local unit

    [ -e "$UNIT_DIR" ] || die "systemd unit directory missing: $UNIT_DIR"
    [ ! -L "$UNIT_DIR" ] ||
        die "systemd unit directory must not be a symlink: $UNIT_DIR"
    [ -d "$UNIT_DIR" ] ||
        die "systemd unit path is not a directory: $UNIT_DIR"

    for unit in "${ALL_UNITS[@]}"; do
        installed_path="$UNIT_DIR/$unit"
        [ -e "$installed_path" ] ||
            die "installed unit missing: $installed_path"
        [ ! -L "$installed_path" ] ||
            die "installed unit must not be a symlink: $installed_path"
        [ -f "$installed_path" ] ||
            die "installed unit is not a regular file: $installed_path"
        [ -r "$installed_path" ] ||
            die "installed unit is not readable: $installed_path"
        installed_mode=$(/usr/bin/stat -c '%a' "$installed_path")
        [ "$installed_mode" = "644" ] ||
            die "installed unit mode must be 0644: $unit (found $installed_mode)"
        if [ "$CRON_ROOT" = "/" ]; then
            installed_owner=$(/usr/bin/stat -c '%U' "$installed_path")
            [ "$installed_owner" = "root" ] ||
                die "installed unit owner must be root: $unit (found $installed_owner)"
        fi
        /usr/bin/cmp -s "$REPO/deploy/$unit" "$installed_path" ||
            die "installed unit differs from source: $unit"
        # 템플릿 유닛(name@.service)은 이 systemd 버전에서 비인스턴스명으로
        # show 조회가 안 된다 — 임시 인스턴스명으로 조회하면 FragmentPath 가
        # 템플릿 파일 경로로 해석된다.
        query_unit="$unit"
        case "$unit" in
        *"@."*)
            query_unit="${unit/@./@fragmentcheck.}"
            ;;
        esac
        if ! fragment_path=$(
            "$SYSTEMCTL_BIN" show "$query_unit" \
                --property=FragmentPath --value
        ); then
            die "cannot resolve systemd FragmentPath for $unit"
        fi
        [ "$fragment_path" = "$installed_path" ] ||
            die "systemd FragmentPath mismatch for $unit: ${fragment_path:-<empty>}"
    done

    for unit in "${TIMER_UNITS[@]}"; do
        "$SYSTEMCTL_BIN" is-enabled --quiet "$unit" ||
            die "timer is not enabled: $unit"
        "$SYSTEMCTL_BIN" is-active --quiet "$unit" ||
            die "timer is not active: $unit"
    done
}

reject_legacy_cron() {
    local cron_file
    local cron_dir
    local cron_files=()

    [ -d "$CRON_ROOT" ] ||
        die "cron inspection root missing; refusing fail-open: $CRON_ROOT"

    for cron_file in \
        "$(root_path /etc/crontab)" \
        "$(root_path /etc/anacrontab)"; do
        [ ! -e "$cron_file" ] || cron_files+=("$cron_file")
    done

    for cron_dir in \
        "$(root_path /etc/cron.d)" \
        "$(root_path /etc/cron.hourly)" \
        "$(root_path /etc/cron.daily)" \
        "$(root_path /etc/cron.weekly)" \
        "$(root_path /etc/cron.monthly)" \
        "$(root_path /var/spool/cron)" \
        "$(root_path /var/spool/cron/crontabs)"; do
        [ ! -d "$cron_dir" ] || { [ -r "$cron_dir" ] && [ -x "$cron_dir" ]; } ||
            die "cannot inspect cron directory; refusing fail-open: $cron_dir"
        if [ -d "$cron_dir" ]; then
            shopt -s nullglob
            for cron_file in "$cron_dir"/*; do
                [ -f "$cron_file" ] && cron_files+=("$cron_file")
            done
            shopt -u nullglob
        fi
    done

    for cron_file in "${cron_files[@]}"; do
        scan_cron_file "$cron_file"
    done
}

reject_duplicate_systemd_schedules() {
    local candidate
    local path
    local listing
    local service_match

    # A disabled legacy unit can later be re-enabled accidentally.  Require the
    # installed prelude timer namespace to contain exactly the supported set.
    if [ -d "$UNIT_DIR" ]; then
        [ -r "$UNIT_DIR" ] && [ -x "$UNIT_DIR" ] ||
            die "cannot inspect unit directory; refusing fail-open: $UNIT_DIR"
        shopt -s nullglob
        for path in "$UNIT_DIR"/prelude-*.timer; do
            candidate=$(basename "$path")
            is_expected_timer "$candidate" ||
                die "unexpected installed prelude timer: $path"
        done
        for path in "$UNIT_DIR"/*.service; do
            candidate=$(basename "$path")
            is_expected_service "$candidate" && continue
            if ! service_match=$(
                /usr/bin/awk '
                    /^[[:space:]]*(ExecStart|ExecStartPre|ExecStartPost)=/ &&
                    (/prelude/ ||
                     /daily_run_(preopen|distribution)\.sh/ ||
                     /daily_close_(preopen|distribution)\.sh/ ||
                     /publish_dashboard\.sh/ ||
                     /backup_db\.sh/ ||
                     /heartbeat\.sh/) { print; exit }
                ' "$path"
            ); then
                die "cannot inspect installed service: $path"
            fi
            [ -z "$service_match" ] ||
                die "unexpected service can duplicate prelude jobs: $path"
        done
        shopt -u nullglob
    fi

    [ -x "$SYSTEMCTL_BIN" ] || die "systemctl unavailable: $SYSTEMCTL_BIN"
    if ! listing=$(
        "$SYSTEMCTL_BIN" list-unit-files 'prelude-*.timer' \
            --no-legend --no-pager 2>&1
    ); then
        echo "$listing" >&2
        die "cannot enumerate installed timer files; refusing fail-open"
    fi
    while read -r candidate _; do
        [ -z "$candidate" ] && continue
        [[ "$candidate" = prelude-*.timer ]] || continue
        is_expected_timer "$candidate" ||
            die "unexpected registered prelude timer: $candidate"
    done <<< "$listing"

    if ! listing=$(
        "$SYSTEMCTL_BIN" list-units --type=timer --all \
            'prelude-*.timer' --no-legend --no-pager --plain 2>&1
    ); then
        echo "$listing" >&2
        die "cannot enumerate active timer jobs; refusing fail-open"
    fi
    while read -r candidate _; do
        [ -z "$candidate" ] && continue
        [[ "$candidate" = prelude-*.timer ]] || continue
        is_expected_timer "$candidate" ||
            die "unexpected loaded prelude timer: $candidate"
    done <<< "$listing"
}

validate_repo_contract
reject_legacy_cron
reject_duplicate_systemd_schedules

if [ "$CHECK_ONLY" -eq 1 ]; then
    validate_installed_contract
    echo "prelude systemd preflight: OK"
    exit 0
fi

[ "$EUID" -eq 0 ] || [ "$FIXTURE_INSTALL" -eq 1 ] ||
    die "run installer as root (sudo)"
[ -e "$UNIT_DIR" ] || die "systemd unit directory missing: $UNIT_DIR"
[ ! -L "$UNIT_DIR" ] ||
    die "systemd unit directory must not be a symlink: $UNIT_DIR"
[ -d "$UNIT_DIR" ] || die "systemd unit path is not a directory: $UNIT_DIR"
exec {INSTALL_LOCK_FD}<"$UNIT_DIR" ||
    die "cannot open systemd unit directory for install locking"
/usr/bin/flock -n "$INSTALL_LOCK_FD" ||
    die "another prelude systemd install is already running"

echo "Installing prelude systemd units..."

STAGE_DIR=""
BACKUP_DIR=""
TRANSACTION_ACTIVE=0
ROLLBACK_FAILED=0
declare -A HAD_ORIGINAL=()
declare -A PREVIOUS_ACTIVE_STATE=()
declare -A PREVIOUS_ENABLED_STATE=()

cleanup_temp_tree() {
    local candidate="$1"
    local kind="$2"

    [ -n "$candidate" ] || return 0
    case "$candidate" in
        "$UNIT_DIR"/.prelude-"$kind".*) ;;
        *)
            echo "WARNING: refusing unsafe temporary cleanup: $candidate" >&2
            return 1
            ;;
    esac
    if [ -e "$candidate" ]; then
        /usr/bin/rm -rf -- "$candidate" || {
            echo "WARNING: temporary cleanup failed: $candidate" >&2
            return 1
        }
    fi
    return 0
}

rollback_install() {
    local unit

    echo "ERROR: install failed; restoring previous unit files and timer state" >&2
    set +e
    # Quiesce definitions from the failed transaction while they still exist,
    # so a first-time install cannot leave active jobs or dangling enablement.
    for unit in "${TIMER_UNITS[@]}"; do
        "$SYSTEMCTL_BIN" stop "$unit" || {
            [ "${PREVIOUS_ACTIVE_STATE[$unit]:-unknown}" = "unknown" ] ||
                ROLLBACK_FAILED=1
        }
        "$SYSTEMCTL_BIN" disable "$unit" || ROLLBACK_FAILED=1
    done

    for unit in "${ALL_UNITS[@]}"; do
        if [ "${HAD_ORIGINAL[$unit]:-0}" -eq 1 ]; then
            /usr/bin/mv -f -- "$BACKUP_DIR/$unit" "$UNIT_DIR/$unit" ||
                ROLLBACK_FAILED=1
        else
            /usr/bin/rm -f -- "$UNIT_DIR/$unit" ||
                ROLLBACK_FAILED=1
        fi
    done

    "$SYSTEMCTL_BIN" daemon-reload || ROLLBACK_FAILED=1
    for unit in "${TIMER_UNITS[@]}"; do
        case "${PREVIOUS_ENABLED_STATE[$unit]:-not-found}" in
            enabled)
                "$SYSTEMCTL_BIN" enable "$unit" || ROLLBACK_FAILED=1
                ;;
            disabled)
                "$SYSTEMCTL_BIN" disable "$unit" || ROLLBACK_FAILED=1
                ;;
            not-found) ;;
        esac
        case "${PREVIOUS_ACTIVE_STATE[$unit]:-unknown}" in
            active)
                "$SYSTEMCTL_BIN" restart "$unit" || ROLLBACK_FAILED=1
                ;;
            inactive)
                "$SYSTEMCTL_BIN" stop "$unit" || ROLLBACK_FAILED=1
                ;;
            unknown) ;;
        esac
    done
    set -e

    if [ "$ROLLBACK_FAILED" -ne 0 ]; then
        echo "ERROR: rollback was incomplete; preserved backup: $BACKUP_DIR" >&2
    else
        echo "Previous systemd configuration restored." >&2
    fi
}

snapshot_timer_state() {
    local enabled_state
    local active_state
    local unit="$1"

    if ! enabled_state=$("$SYSTEMCTL_BIN" is-enabled "$unit" 2>&1); then
        case "$enabled_state" in
            disabled|not-found) ;;
            "Failed to get unit file state for $unit: No such file or directory")
                # 신규 유닛 최초 설치: 이 systemd 는 not-found 토큰 대신 이
                # 문구를 stderr 로 낸다.  정확히 이 유닛의 이 문구만 관용.
                enabled_state="not-found"
                ;;
            *)
                die "cannot determine prior enabled state for $unit: ${enabled_state:-<empty>}"
                ;;
        esac
    fi
    case "$enabled_state" in
        enabled)
            PREVIOUS_ENABLED_STATE["$unit"]="enabled"
            ;;
        disabled|not-found)
            PREVIOUS_ENABLED_STATE["$unit"]="$enabled_state"
            ;;
        *)
            die "unsupported prior enabled state for $unit: ${enabled_state:-<empty>}"
            ;;
    esac

    # 존재하지 않던 신규 timer는 active 상태도 정의되지 않는다. 일부
    # systemd는 is-active를 "inactive"로 답하지만 이를 기억하면 rollback이
    # 파일 제거 후 존재하지 않는 unit에 stop을 호출해 복구를 거짓 실패시킨다.
    if [ "${PREVIOUS_ENABLED_STATE[$unit]}" = "not-found" ]; then
        PREVIOUS_ACTIVE_STATE["$unit"]="unknown"
        return
    fi

    if ! active_state=$("$SYSTEMCTL_BIN" is-active "$unit" 2>&1); then
        case "$active_state" in
            inactive|unknown) ;;
            *)
                die "cannot determine prior active state for $unit: ${active_state:-<empty>}"
                ;;
        esac
    fi
    case "$active_state" in
        active)
            PREVIOUS_ACTIVE_STATE["$unit"]="active"
            ;;
        inactive|unknown)
            PREVIOUS_ACTIVE_STATE["$unit"]="$active_state"
            ;;
        *)
            die "unsupported prior active state for $unit: ${active_state:-<empty>}"
            ;;
    esac
}

finish_install() {
    local status=$?

    trap - EXIT HUP INT TERM
    if [ "$status" -ne 0 ] && [ "$TRANSACTION_ACTIVE" -eq 1 ]; then
        rollback_install
    fi
    cleanup_temp_tree "$STAGE_DIR" install || true
    if [ "$ROLLBACK_FAILED" -eq 0 ]; then
        cleanup_temp_tree "$BACKUP_DIR" backup || true
    fi
    exit "$status"
}
trap finish_install EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

STAGE_DIR=$(/usr/bin/mktemp -d "$UNIT_DIR/.prelude-install.XXXXXX")
BACKUP_DIR=$(/usr/bin/mktemp -d "$UNIT_DIR/.prelude-backup.XXXXXX")
stage_paths=()
for unit in "${ALL_UNITS[@]}"; do
    /usr/bin/install -m 0644 "$REPO/deploy/$unit" "$STAGE_DIR/$unit"
    /usr/bin/cmp -s "$REPO/deploy/$unit" "$STAGE_DIR/$unit" ||
        die "staged unit differs from source: $unit"
    stage_paths+=("$STAGE_DIR/$unit")
done
"$SYSTEMD_ANALYZE_BIN" verify "${stage_paths[@]}" ||
    die "systemd-analyze rejected one or more staged units"

for unit in "${TIMER_UNITS[@]}"; do
    snapshot_timer_state "$unit"
done

for unit in "${ALL_UNITS[@]}"; do
    installed_path="$UNIT_DIR/$unit"
    if [ -e "$installed_path" ] || [ -L "$installed_path" ]; then
        [ ! -L "$installed_path" ] ||
            die "refusing to replace symlinked installed unit: $installed_path"
        [ -f "$installed_path" ] ||
            die "refusing to replace non-regular installed unit: $installed_path"
        /usr/bin/cp -p -- "$installed_path" "$BACKUP_DIR/$unit"
        /usr/bin/cmp -s "$installed_path" "$BACKUP_DIR/$unit" ||
            die "unit backup differs from installed file: $unit"
        HAD_ORIGINAL["$unit"]=1
    else
        HAD_ORIGINAL["$unit"]=0
    fi
done

TRANSACTION_ACTIVE=1
for unit in "${ALL_UNITS[@]}"; do
    /usr/bin/mv -f -- "$STAGE_DIR/$unit" "$UNIT_DIR/$unit"
done

"$SYSTEMCTL_BIN" daemon-reload
"$SYSTEMCTL_BIN" enable "${TIMER_UNITS[@]}"
# daemon-reload updates definitions, but an already-active timer may retain its
# old next elapse. Restart re-arms all eight calendars; Persistent=true timers
# may then perform their intended missed-run catch-up.
"$SYSTEMCTL_BIN" restart "${TIMER_UNITS[@]}"
validate_installed_contract

echo
echo "=== Installed timers ==="
"$SYSTEMCTL_BIN" list-timers --no-pager "${TIMER_UNITS[@]}"
TRANSACTION_ACTIVE=0
cleanup_temp_tree "$STAGE_DIR" install ||
    die "could not remove completed install staging directory"
STAGE_DIR=""
cleanup_temp_tree "$BACKUP_DIR" backup ||
    die "could not remove completed install backup directory"
BACKUP_DIR=""

echo
echo "=== Test ==="
echo "  sudo systemctl start prelude-distribution.service  # 수동 1회 실행"
echo "  systemctl status prelude-distribution.timer        # 다음 실행 시각 확인"
echo "  journalctl -u prelude-distribution.service -n 50   # 로그 확인"
echo
echo "install complete: no legacy cron or duplicate prelude timer detected."
