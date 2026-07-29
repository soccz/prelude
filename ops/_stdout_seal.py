"""fd 수준 stdout 봉인 — close gate NUL 프로토콜 절대 방어 (2026-07-29).

로거 stderr 고정(propagate=False)과 ``_force_stderr_logging()`` 은 logging
계층 방어일 뿐이다.  같은 오염 클래스(모듈이 root logging 을 stdout 으로
구성 → 셸이 기계 파싱하는 stdout 에 경고가 섞임)가 07-28 gate,
07-29 snapshot/labels 로 이틀 연속 다른 모듈에서 재발했다 — 새 모듈이
추가될 때마다 뚫릴 수 있는 방어는 방어가 아니다.

여기서는 OS 수준에서 닫는다: close gate 가 NUL 프로토콜 ``__main__`` 으로
기동될 때(다른 prelude 모듈 import 보다 먼저) fd1 을 프로토콜 전용 fd 로
복제해 두고, fd1 자체를 stderr 로 바꿔치기한다. 이후 이 프로세스의 어떤
코드가 print()/logging/C-level write 로 stdout 에 무엇을 쓰든 셸 파서에는
닿을 수 없고, 프로토콜 레코드만 보존된 fd 로 나간다.

이 모듈 자체에는 import 부작용이 없다. ``ops.close_input_gate`` 가 실제
entrypoint일 때만 ``seal_nul_protocol_stdout`` 을 호출해야 한다. 그래야
gate를 라이브러리로 import하는 다른 CLI의 stdout을 오봉인하지 않는다.
"""
from __future__ import annotations

import os
import stat
import sys
from typing import BinaryIO


def _wants_nul_protocol(argv: list[str]) -> bool:
    if "--help" in argv or "-h" in argv:
        return False
    output_format: str | None = None
    for index, arg in enumerate(argv):
        if arg == "--output-format" and index + 1 < len(argv):
            output_format = argv[index + 1]
        elif arg.startswith("--output-format="):
            output_format = arg.split("=", 1)[1]
    # argparse의 반복 option 계약과 동일하게 마지막 값을 사용한다.
    return output_format == "nul"


def seal_nul_protocol_stdout(argv: list[str]) -> BinaryIO | None:
    if not _wants_nul_protocol(argv):
        return None
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    stdout_stat = os.fstat(stdout_fd)
    stderr_stat = os.fstat(stderr_fd)
    stdout_target = (
        stdout_stat.st_dev,
        stdout_stat.st_ino,
        stat.S_IFMT(stdout_stat.st_mode),
    )
    stderr_target = (
        stderr_stat.st_dev,
        stderr_stat.st_ino,
        stat.S_IFMT(stderr_stat.st_mode),
    )
    if stdout_target == stderr_target:
        raise RuntimeError(
            "NUL protocol requires distinct stdout and stderr destinations"
        )
    sys.stdout.flush()
    protocol_fd = os.dup(stdout_fd)
    os.set_inheritable(protocol_fd, False)
    os.dup2(stderr_fd, stdout_fd)
    return os.fdopen(protocol_fd, "wb")
