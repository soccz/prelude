"""NUL 프로토콜 stdout 봉인의 적대적 e2e 검증.

07-28/29 사고 클래스 재현: root logging 이 stdout 으로 구성된 프로세스에서
print()·logging 오염이 발생해도, 셸이 기계 파싱하는 stdout 에는 프로토콜
레코드만 남아야 한다.  logging 계층 방어(모듈별 stderr 고정)가 아니라
fd 봉인(ops/_stdout_seal.py)이 주는 클래스 수준 보증을 서브프로세스로
직접 확인한다.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ops._stdout_seal import _wants_nul_protocol

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SENTINEL = "__PRELUDE_CLOSE_PLAN_V1_OK__"

# 적대 조건: 07-28 사고처럼 gate import 전에 root logging 이 stdout 을
# 물고 있고, main 종료 후(atexit)에도 print/logging 오염이 발생한다.
HOSTILE_WRAPPER = textwrap.dedent(
    """
    import atexit
    import logging
    import os
    import sys

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    def _pollute():
        print("HOSTILE-PRINT-POLLUTION")
        os.write(1, b"HOSTILE-FD1-POLLUTION\\n")
        logging.getLogger("hostile.module").warning(
            "HOSTILE-LOG-POLLUTION"
        )

    atexit.register(_pollute)

    import runpy

    runpy.run_module("ops.close_input_gate", run_name="__main__")
    """
)


def test_nul_stdout_survives_hostile_logging_and_prints(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            HOSTILE_WRAPPER,
            "--through-asof",
            "2026-07-26",
            "--cohort",
            "pump-v1",
            "--output-root",
            str(tmp_path / "output"),
            "--output-format",
            "nul",
        ],
        capture_output=True,
        cwd=PROJECT_ROOT,
        timeout=120,
    )
    stderr_text = proc.stderr.decode("utf-8", errors="replace")

    assert proc.returncode == 0, stderr_text

    # stdout 은 순수 NUL 레코드 스트림이어야 한다.
    assert proc.stdout.endswith(b"\0"), proc.stdout
    records = proc.stdout.decode("utf-8").split("\0")[:-1]
    assert records[-1] == SENTINEL
    body = records[:-1]
    assert body and len(body) % 2 == 0, records
    for decision_date, mode in zip(body[::2], body[1::2]):
        assert decision_date == "2026-07-26"
        assert mode in {
            "close",
            "skip-zero-pick",
            "skip-legacy-unverifiable",
            "skip-no-decision",
            "skip-terminal-kill",
            "skip-policy-blocked",
        }, records

    # 오염은 전부 stderr 로 밀려나야 한다.
    assert b"HOSTILE" not in proc.stdout
    assert "HOSTILE-PRINT-POLLUTION" in stderr_text
    assert "HOSTILE-FD1-POLLUTION" in stderr_text
    assert "HOSTILE-LOG-POLLUTION" in stderr_text


def test_real_bash_coproc_mapfile_receives_only_protocol_records(tmp_path):
    """Production과 같은 coproc/mapfile 소비면까지 포함한 통합 회귀."""
    stderr_path = tmp_path / "gate.stderr"
    output_root = tmp_path / "output"
    shell = textwrap.dedent(
        r"""
        set -uo pipefail
        coproc PRELUDE_CLOSE_PLAN {
            "$1" -c "$2" \
                --through-asof 2026-07-26 \
                --cohort pump-v1 \
                --output-root "$3" \
                --output-format nul 2>"$4"
        }
        plan_fd="${PRELUDE_CLOSE_PLAN[0]}"
        plan_pid="$PRELUDE_CLOSE_PLAN_PID"
        records=()
        if ! mapfile -d '' -t records <&"$plan_fd"; then
            records=()
        fi
        if wait "$plan_pid"; then
            :
        else
            plan_rc=$?
            exit "$plan_rc"
        fi
        printf '%s\n' "${records[@]}"
        """
    )

    proc = subprocess.run(
        [
            "/bin/bash",
            "-c",
            shell,
            "prelude-close-test",
            sys.executable,
            HOSTILE_WRAPPER,
            str(output_root),
            str(stderr_path),
        ],
        capture_output=True,
        cwd=PROJECT_ROOT,
        timeout=120,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.splitlines() == [
        "2026-07-26",
        "skip-legacy-unverifiable",
        SENTINEL,
    ]
    stderr_text = stderr_path.read_text(encoding="utf-8")
    assert "HOSTILE-PRINT-POLLUTION" in stderr_text
    assert "HOSTILE-FD1-POLLUTION" in stderr_text
    assert "HOSTILE-LOG-POLLUTION" in stderr_text


def test_text_mode_stdout_is_not_sealed(tmp_path):
    """text 모드(사람/헬스체크용)는 봉인 대상이 아니다 — 기존 계약 유지."""
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "ops.close_input_gate",
            "--asof",
            "2026-07-26",
            "--cohort",
            "pump-v1",
            "--output-root",
            str(tmp_path / "output"),
        ],
        capture_output=True,
        cwd=PROJECT_ROOT,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    assert proc.stdout.decode().strip() == "skip-legacy-unverifiable"


@pytest.mark.parametrize(
    "module",
    (
        "ops.policy_competition",
        "ops.v2_provenance",
        "scripts.v2_scoreboard",
        "scripts.close_recommend_ledger",
    ),
)
def test_gate_importers_do_not_seal_their_own_stdout(module):
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            f'import {module}; print("MACHINE-OK")',
            "--output-format",
            "nul",
        ],
        capture_output=True,
        cwd=PROJECT_ROOT,
        timeout=120,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == "MACHINE-OK\n"


def test_repeated_output_format_uses_argparse_last_value(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "ops.close_input_gate",
            "--asof",
            "2026-07-26",
            "--cohort",
            "pump-v1",
            "--output-root",
            str(tmp_path / "output"),
            "--output-format",
            "nul",
            "--output-format",
            "text",
        ],
        capture_output=True,
        cwd=PROJECT_ROOT,
        timeout=120,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "skip-legacy-unverifiable"


def test_help_remains_human_stdout_even_with_nul_flag():
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "ops.close_input_gate",
            "--output-format",
            "nul",
            "--help",
        ],
        capture_output=True,
        cwd=PROJECT_ROOT,
        timeout=120,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    assert "usage:" in proc.stdout


def test_abbreviated_output_format_is_rejected_before_unsealed_protocol():
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            HOSTILE_WRAPPER,
            "--through-asof",
            "2026-07-26",
            "--cohort",
            "pump-v1",
            "--output-f",
            "nul",
        ],
        capture_output=True,
        cwd=PROJECT_ROOT,
        timeout=120,
    )

    assert proc.returncode != 0
    assert SENTINEL.encode() not in proc.stdout
    assert b"unrecognized arguments: --output-f nul" in proc.stderr


def test_nul_protocol_rejects_merged_stdout_and_stderr():
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "ops.close_input_gate",
            "--through-asof",
            "2026-07-26",
            "--cohort",
            "pump-v1",
            "--output-format",
            "nul",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=PROJECT_ROOT,
        timeout=120,
    )

    assert proc.returncode != 0
    assert SENTINEL.encode() not in proc.stdout
    assert b"requires distinct stdout and stderr" in proc.stdout


def test_wants_nul_protocol_matches_cli_forms():
    assert _wants_nul_protocol(["--output-format", "nul"])
    assert _wants_nul_protocol(["--output-format=nul"])
    assert _wants_nul_protocol(
        ["--through-asof", "2026-07-26", "--output-format", "nul"]
    )
    assert not _wants_nul_protocol([])
    assert not _wants_nul_protocol(["--output-format", "text"])
    assert not _wants_nul_protocol(
        ["--output-format", "nul", "--output-format", "text"]
    )
    assert not _wants_nul_protocol(["--output-format", "nul", "--help"])
    assert not _wants_nul_protocol(["--output-format"])
    assert not _wants_nul_protocol(["nul"])
