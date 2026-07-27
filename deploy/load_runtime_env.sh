#!/usr/bin/env bash
# shellcheck shell=bash
#
# Import the strictly parsed project .env without evaluating it as shell code.
# The Python parser validates the complete file before emitting any record.

load_prelude_runtime_env() {
    local env_file="$1"
    local python_bin="$2"
    local export_fd
    local parser_pid
    local parser_rc
    local sentinel="__PRELUDE_RUNTIME_ENV_V1_OK__"
    local key
    local last_index
    local value
    local index
    local -a records=()

    if [ ! -e "$env_file" ] && [ ! -L "$env_file" ]; then
        return 0
    fi
    if [ ! -x "$python_bin" ]; then
        echo "runtime environment parser Python is unavailable: $python_bin" >&2
        return 2
    fi

    coproc PRELUDE_RUNTIME_ENV_EXPORT {
        "$python_bin" -m ops.runtime_env "$env_file" --export-nul
    }
    export_fd="${PRELUDE_RUNTIME_ENV_EXPORT[0]}"
    parser_pid="$PRELUDE_RUNTIME_ENV_EXPORT_PID"
    if ! mapfile -d '' -t records <&"$export_fd"; then
        records=()
    fi
    if wait "$parser_pid"; then
        parser_rc=0
    else
        parser_rc=$?
    fi
    unset PRELUDE_RUNTIME_ENV_EXPORT PRELUDE_RUNTIME_ENV_EXPORT_PID
    if [ "$parser_rc" -ne 0 ] ||
       [ "${#records[@]}" -lt 1 ] ||
       [ "${records[${#records[@]} - 1]}" != "$sentinel" ]; then
        echo "runtime environment validation/export failed: $env_file" >&2
        return 2
    fi
    last_index=$((${#records[@]} - 1))
    unset "records[$last_index]"
    if [ $(( ${#records[@]} % 2 )) -ne 0 ]; then
        echo "runtime environment parser returned malformed records" >&2
        return 2
    fi

    for ((index = 0; index < ${#records[@]}; index += 2)); do
        key="${records[index]}"
        value="${records[index + 1]}"
        case "$key" in
            PRELUDE_DASHBOARD_PIN|TELEGRAM_BOT_TOKEN|TELEGRAM_CHAT_ID) ;;
            *)
                echo "runtime environment parser returned unsafe key: $key" >&2
                return 2
                ;;
        esac
        printf -v "$key" '%s' "$value"
        export "$key"
    done
}
