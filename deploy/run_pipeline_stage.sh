#!/usr/bin/env bash
# Serialize post-open jobs that read and rewrite the same ledgers/artifacts.
#
# The lock covers the whole child process.  For publish, the latest *current
# KST-day* close sessions must also have completed successfully; publishing
# yesterday's/stale artifacts is worse than a loud failed unit.

set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "usage: $0 <distribution-close|preopen-close|publish|heartbeat> <command> [args...]" >&2
    exit 64
fi

STAGE="$1"
shift

case "$STAGE" in
    distribution-close|preopen-close|publish|heartbeat) ;;
    *)
        echo "ERROR: unsupported pipeline stage: $STAGE" >&2
        exit 64
        ;;
esac

PROJ_ROOT="${PRELUDE_ROOT:-/home/soccz/22tb/prelude}"
OUTPUT_DIR="$PROJ_ROOT/output"
STATE_DIR="$OUTPUT_DIR/pipeline_state"
mkdir -p "$OUTPUT_DIR"

SCRIPT_SELF=$(/usr/bin/realpath -e -- "${BASH_SOURCE[0]}")
LOCK_EXEC="$(dirname "$SCRIPT_SELF")/lock_exec.py"
MARKER_EXEC="$(dirname "$SCRIPT_SELF")/pipeline_marker.py"
LOCK_FILE="$OUTPUT_DIR/.postopen-pipeline.lock"

# Never redirect to a lock pathname: shell redirection truncates regular files
# and follows symlinks before flock(2) can protect anything.  The helper keeps
# a no-follow, inode-validated descriptor alive for the whole recursive child.
if [[ -v PRELUDE_POSTOPEN_LOCK_FD ]]; then
    /usr/bin/python3 "$LOCK_EXEC" verify \
        --lock-file "$LOCK_FILE" \
        --fd "$PRELUDE_POSTOPEN_LOCK_FD"
else
    exec /usr/bin/python3 "$LOCK_EXEC" run \
        --lock-file "$LOCK_FILE" \
        --fd-env PRELUDE_POSTOPEN_LOCK_FD \
        --wait \
        --busy-exit 75 \
        --busy-message \
        "ERROR: post-open pipeline is already active; retry required" \
        -- /bin/bash "$SCRIPT_SELF" "$STAGE" "$@"
fi

latest_session_succeeded() {
    local log_path="$1"
    local session_prefix="$2"
    local expected_header="$3"
    local header_line
    local latest_header
    local last_done

    if [ ! -s "$log_path" ]; then
        echo "ERROR: required close log missing or empty: $log_path" >&2
        return 1
    fi

    header_line=$(
        /usr/bin/awk -v prefix="$session_prefix" \
            'index($0, prefix) == 1 { line = NR } END { if (line) print line }' \
            "$log_path"
    )
    if [ -z "$header_line" ]; then
        echo "ERROR: current-KST-day close session missing: $log_path" >&2
        return 1
    fi

    latest_header=$(/usr/bin/sed -n "${header_line}p" "$log_path")
    if [[ "$latest_header" != "$expected_header"* ]]; then
        echo "ERROR: latest close session is not for the current KST day: $log_path" >&2
        return 1
    fi

    last_done=$(
        /usr/bin/tail -n "+$header_line" "$log_path" |
            /usr/bin/awk '/^\[done\] / { line = $0 } END { print line }'
    )
    if [[ ! "$last_done" =~ ^\[done\]\ [[:digit:]]{2}:[[:digit:]]{2}:[[:digit:]]{2}\ exit=0$ ]]; then
        echo "ERROR: latest close session did not finish successfully: $log_path" >&2
        return 1
    fi
}

marker_succeeded() {
    local marker_path="$1"
    local expected_stage="$2"
    local expected_date="$3"

    if [ ! -e "$marker_path" ] && [ ! -L "$marker_path" ]; then
        echo "ERROR: current close success marker missing: $marker_path" >&2
        return 1
    fi
    if ! /usr/bin/python3 "$MARKER_EXEC" verify \
        --marker "$marker_path" \
        --stage "$expected_stage" \
        --kst-date "$expected_date"; then
        echo "ERROR: invalid current close success marker: $marker_path" >&2
        return 1
    fi
}

read -r TODAY_COMPACT TODAY_ISO < <(
    TZ=Asia/Seoul /usr/bin/date '+%Y%m%d %F'
)

if [ "$STAGE" = "distribution-close" ] || [ "$STAGE" = "preopen-close" ]; then
    mkdir -p "$STATE_DIR"
    MARKER="$STATE_DIR/${STAGE}_${TODAY_COMPACT}.ok"
    /usr/bin/python3 "$MARKER_EXEC" clear --marker "$MARKER"

    if "$@"; then
        COMPLETED_AT=$(
            TZ=Asia/Seoul /usr/bin/date --iso-8601=seconds
        )
        /usr/bin/python3 "$MARKER_EXEC" write \
            --marker "$MARKER" \
            --stage "$STAGE" \
            --kst-date "$TODAY_ISO" \
            --completed-at "$COMPLETED_AT"
        exit 0
    else
        CHILD_RC=$?
        exit "$CHILD_RC"
    fi
fi

if [ "$STAGE" = "publish" ]; then
    marker_succeeded \
        "$STATE_DIR/distribution-close_${TODAY_COMPACT}.ok" \
        "distribution-close" \
        "$TODAY_ISO"
    marker_succeeded \
        "$STATE_DIR/preopen-close_${TODAY_COMPACT}.ok" \
        "preopen-close" \
        "$TODAY_ISO"
    latest_session_succeeded \
        "$OUTPUT_DIR/cron_close_${TODAY_COMPACT}.log" \
        "=== prelude distribution close KST " \
        "=== prelude distribution close KST ${TODAY_ISO} "
    latest_session_succeeded \
        "$OUTPUT_DIR/cron_preopen_close_${TODAY_COMPACT}.log" \
        "=== prelude pre-open close KST " \
        "=== prelude pre-open close KST ${TODAY_ISO} "
fi

exec "$@"
